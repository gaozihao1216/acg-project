# -*- coding: utf-8 -*-
"""重放所有案例，逐字节校验产物——把"可复现"从声明变成断言。

为什么需要它
------------
`input/request.json` 里写着 `command`、写着随机种子、写着凸包面数上限，
但那只是**一句话**。没有任何东西证明"照它重放能拿回同一份产物"。

这个脚本就是那句证明：

    results/w3/
    ├── prism/input/request.json  → 重放 → 与 output/prism.asset.json 逐字节比
    ├── shell/input/request.json  → 重放 → …
    ├── cloud/input/request.json  → 重放 → …（随机点云，比的是 seed 的确定性）
    └── stack/input/request.json  → 重放 → 与 output/stack.scene.json 逐字节比

它同时是**对资产设计的审查**：只要输入卡漏记了任何一个能改变结果的自由度
（比如 `max_faces`、修正式、质量来源），重放出来的产物就会不同，哈希立刻
对不上。所以它测的不是脚本，而是**规格有没有漏**。

怎么重放
--------
不解析参数、不复制装配逻辑——直接**执行记录下来的那条命令**，只把路径前缀
换到临时目录。这样测的是真实入口，而不是我另写一遍的等价流程（那种等价
流程迟早会和真入口分叉，于是测试永远绿）。

场景案例依赖物体案例，所以按 ``case_request/1`` → ``scene_request/1`` 排序。

用法::

    python Code/scripts/reproduce.py --root results/w3

    # 只查几个
    python Code/scripts/reproduce.py --root results/w3 --only prism stack

    # 规格变了（例如求解器行为改进）→ 刷新产物，然后再校验一遍
    python Code/scripts/reproduce.py --root results/w3 --refresh
    python Code/scripts/reproduce.py --root results/w3
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_CODE_ROOT)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

_LINE = "=" * 78
_SUB = "-" * 78

#: 只复制这些（体积大的 .blend/.png/.mp4 与重放无关）
_COPY_SUFFIXES = (".json", ".obj", ".stl", ".csv", ".txt")
_COPY_LIMIT = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# 发现案例
# ---------------------------------------------------------------------------

def find_cases(root: str, only: list[str] | None = None) -> list[dict]:
    """扫描 ``root/*/input/request.json``，按重放顺序返回案例列表。"""
    cases: list[dict] = []
    for name in sorted(os.listdir(root)):
        case_dir = os.path.join(root, name)
        request_path = os.path.join(case_dir, "input", "request.json")
        if not os.path.isfile(request_path):
            continue
        if only and name not in only:
            continue

        with open(request_path, "r", encoding="utf-8") as handle:
            request = json.load(handle)

        schema = request.get("schema")
        if schema not in ("case_request/1", "scene_request/1"):
            print(f"  跳过 {name}：不认识的输入卡 {schema!r}")
            continue

        # 产物名固定为 <name>.*.json（asset 或 scene）
        output_dir = os.path.join(case_dir, "output")
        produced = sorted(
            entry for entry in os.listdir(output_dir)
            if entry.endswith(".json") and entry.startswith(name + ".")
            and not entry.endswith(".expanded.scene.json")
        ) if os.path.isdir(output_dir) else []

        cases.append({
            "name": name,
            "dir": case_dir,
            "schema": schema,
            "request": request,
            "produced": produced,
        })

    # 物体案例先跑：场景案例引用它们算出的指纹
    cases.sort(key=lambda item: (item["schema"] != "case_request/1", item["name"]))
    return cases


# ---------------------------------------------------------------------------
# 重放
# ---------------------------------------------------------------------------

def copy_case_tree(source_root: str, target_root: str) -> None:
    """复制 ``input/``，**让 ``output/`` 从空开始**。

    这一步的关键是"不复制 output"。第一版我把 output 里的 JSON 也拷过去了，
    于是比对变成了"副本 vs 原件"——永远相等，`stack.poses.json` 明明没被重放
    也报"一致"。那是个假阳性，比不测更糟。

    现在 output 为空，所以"临时目录里存在某个产物"本身就证明是这次重放生成的。
    顺带也不用整棵复制：``results/`` 里有几百 MB 的 .blend/.png/.mp4。

    物体案例排在场景案例之前重放，所以场景解析引用时，物体的 asset.json
    已经由前一步重新生成好了——这也是 ``find_cases`` 要排序的原因。
    """
    for name in os.listdir(source_root):
        source_case = os.path.join(source_root, name)
        if not os.path.isdir(source_case):
            continue

        os.makedirs(os.path.join(target_root, name, "output"), exist_ok=True)

        source_input = os.path.join(source_case, "input")
        if not os.path.isdir(source_input):
            continue
        target_input = os.path.join(target_root, name, "input")
        os.makedirs(target_input, exist_ok=True)
        for entry in os.listdir(source_input):
            source_file = os.path.join(source_input, entry)
            if not os.path.isfile(source_file):
                continue
            if not entry.endswith(_COPY_SUFFIXES):
                continue
            if os.path.getsize(source_file) > _COPY_LIMIT:
                continue
            shutil.copy2(source_file, os.path.join(target_input, entry))


def rewrite_paths(command: list[str], source_root: str, target_root: str) -> list[str]:
    """把命令里指向原案例根目录的路径改写到临时根目录。"""
    source_abs = os.path.normpath(os.path.abspath(source_root))
    target_abs = os.path.normpath(os.path.abspath(target_root))
    prefix = source_abs + os.sep

    rewritten: list[str] = []
    for token in command:
        candidate = token
        # 输入卡里存的是**相对路径**（`results\w3`），必须先用 abspath 归一化
        # 再比前缀。第一版直接拿 normpath 比，相对路径永远匹配不上，
        # 于是重放全写回了真实目录——临时目录空着，四个案例全都"没有产出"。
        if os.sep in token or "/" in token or token.startswith("."):
            normalized = os.path.normpath(os.path.abspath(token))
            if normalized == source_abs:
                candidate = target_abs
            elif normalized.startswith(prefix):
                candidate = os.path.join(target_abs, normalized[len(prefix):])
        rewritten.append(candidate)
    return rewritten


def replay(case: dict, target_root: str) -> tuple[int, str]:
    """执行记录下来的那条命令（路径改写到临时目录）。"""
    command = list(case["request"].get("command") or [])
    if not command:
        return 1, "输入卡里没有 command 字段，无法重放"

    script = os.path.join(_HERE, command[0])
    if not os.path.isfile(script):
        return 1, f"重放脚本不存在：{command[0]}"

    arguments = rewrite_paths(command[1:], os.path.dirname(case["dir"]), target_root)
    # 物体案例会默认调用 Blender 渲染预览图（与数值无关，但要几十秒），
    # 重放时关掉。场景案例本来就不渲染，不能硬塞这个开关——
    # assemble_scene.py 不认它，会直接报错退出。
    if case["schema"] == "case_request/1" and "--no-render" not in arguments:
        arguments.append("--no-render")

    completed = subprocess.run(
        [sys.executable, script] + arguments,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return completed.returncode, tail[-1] if tail else "（无输出）"
    return 0, ""


# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------

def replay_poses(case: dict, target_root: str) -> tuple[bool, str]:
    """场景案例的派生重放：用重放出来的场景资产再跑一遍仿真。

    为什么值得多花 3 秒：它验证的是**物理求解器的确定性**。同一份场景跑两次
    得到逐字节相同的位姿序列，说明没有引入随机源（哈希遍历顺序、未初始化的
    累积器、并行归约的顺序差异）——这类不确定性一旦存在，调试时会被误认为
    "物理不稳定"。

    返回 ``(是否有可比对项, 说明)``；没有 poses 文件时返回 ``(True, "")``。
    """
    name = case["name"]
    original = os.path.join(case["dir"], "output", f"{name}.poses.json")
    if not os.path.isfile(original):
        return True, ""

    scene_asset = os.path.join(target_root, name, "output", f"{name}.scene.json")
    if not os.path.isfile(scene_asset):
        return False, f"      {name}.poses.json: 跳过（场景未重放成功）"

    replayed = os.path.join(target_root, name, "output", f"{name}.poses.json")
    completed = subprocess.run(
        [
            sys.executable,
            os.path.join(_HERE, "run_sim.py"),
            "--scene-asset", scene_asset,
            "--out", replayed,
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return False, f"      {name}.poses.json: 仿真重放失败（{tail[-1] if tail else '无输出'}）"

    same = filecmp.cmp(original, replayed, shallow=False)
    size = os.path.getsize(original)
    detail = f"      {name}.poses.json: {'一致' if same else '**不一致**'}（{size} 字节，仿真确定性）"
    if not same:
        detail += f"\n        {_first_difference(original, replayed)}"
    return same, detail


def refresh_case(case: dict, target_root: str) -> str:
    """用重放结果覆盖真实产物（``--refresh`` 模式）。

    什么时候需要：**规格变了**，旧产物本来就该变。比如加了暖启动之后，``stack``
    的位姿序列必然不同——此时"不一致"不是 bug，而是"产物过期"。手工重跑命令
    容易漏掉 ``*.poses.json``（它由第二条命令产出），所以这里一起处理。

    实现上仍然先重放到临时目录再复制，避免半路失败把真实产物改成半成品。
    """
    name = case["name"]
    output_dir = os.path.join(case["dir"], "output")
    replayed_dir = os.path.join(target_root, name, "output")

    updated = []
    for entry in sorted(os.listdir(replayed_dir)):
        if not entry.endswith(".json"):
            continue
        source = os.path.join(replayed_dir, entry)
        target = os.path.join(output_dir, entry)
        if os.path.isfile(target) and filecmp.cmp(source, target, shallow=False):
            continue
        shutil.copyfile(source, target)
        updated.append(entry)

    lines = [f"      产物：{'更新 ' + '、'.join(updated) if updated else '无需更新'}"]

    # 位姿序列由 run_sim 产出，重放上面那条命令不会生成它，必须单独重跑
    if case["schema"] == "scene_request/1":
        scene_asset = os.path.join(output_dir, f"{name}.scene.json")
        poses = os.path.join(output_dir, f"{name}.poses.json")
        if os.path.isfile(poses):
            completed = subprocess.run(
                [
                    sys.executable,
                    os.path.join(_HERE, "run_sim.py"),
                    "--scene-asset", scene_asset,
                    "--out", poses,
                ],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            lines.append(
                "      位姿：重新生成" if completed.returncode == 0
                else f"      位姿：失败（{(completed.stderr or '').strip().splitlines()[-1:] or '无输出'}）"
            )

    return "\n".join(lines)


def compare(case: dict, target_root: str) -> tuple[bool, str]:
    """逐文件比对重放产物与原产物。"""
    if not case["produced"]:
        return False, "output/ 里没有可比对的 JSON 产物"

    # 主产物必须有且一致；其余（引擎快照、位姿序列）由别的命令产出，
    # 重放这一步不生成它们，只作提示，不算失败。
    primary = (
        f"{case['name']}.asset.json"
        if case["schema"] == "case_request/1"
        else f"{case['name']}.scene.json"
    )

    lines: list[str] = []
    ok = True
    for entry in case["produced"]:
        original = os.path.join(case["dir"], "output", entry)
        replayed = os.path.join(target_root, case["name"], "output", entry)

        if not os.path.isfile(replayed):
            if entry == primary:
                ok = False
                lines.append(f"      {entry}: **重放没有产出**")
            elif entry.endswith(".poses.json"):
                # 由 replay_poses 单独重放（需要先有场景），这里不重复报告
                continue
            else:
                lines.append(f"      {entry}: 跳过（由其它命令产出）")
            continue

        same = filecmp.cmp(original, replayed, shallow=False)
        ok = ok and same
        size = os.path.getsize(original)
        lines.append(
            f"      {entry}: {'一致' if same else '**不一致**'}（{size} 字节）"
        )
        if not same:
            lines.append(f"        {_first_difference(original, replayed)}")

    return ok, "\n".join(lines)


def _first_difference(left_path: str, right_path: str) -> str:
    """指出第一个不同的行，否则只能说"不一致"——那等于没给线索。"""
    with open(left_path, "r", encoding="utf-8") as handle:
        left = handle.read().splitlines()
    with open(right_path, "r", encoding="utf-8") as handle:
        right = handle.read().splitlines()

    for index, (a, b) in enumerate(zip(left, right), 1):
        if a != b:
            return f"第一个不同的行 {index}:\n          原: {a.strip()[:100]}\n          新: {b.strip()[:100]}"
    return f"行数不同：原 {len(left)} 行，重放 {len(right)} 行"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="重放所有案例并逐字节校验产物",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True, help="案例根目录（含 */input/request.json）")
    parser.add_argument("--only", nargs="+", default=None, help="只查这些案例")
    parser.add_argument("--keep", action="store_true", help="保留临时目录以便排查")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="用重放结果覆盖真实产物（规格变更后使用，不做比对）",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"错误：找不到目录 {args.root}", file=sys.stderr)
        return 2

    cases = find_cases(args.root, args.only)
    print(_LINE)
    print(f"重放校验：{args.root}")
    print(_LINE)
    if not cases:
        print("  没有找到任何 input/request.json")
        return 2

    for case in cases:
        print(
            f"  {case['name']:<10} {case['schema']:<16} "
            f"产物 {len(case['produced'])} 个"
        )

    temporary = tempfile.mkdtemp(prefix="reproduce_")
    target_root = os.path.join(temporary, os.path.basename(os.path.abspath(args.root)))
    copy_case_tree(args.root, target_root)

    passed = 0
    try:
        for case in cases:
            print()
            print(f"[{case['name']}]")
            print(_SUB)
            code, message = replay(case, target_root)
            if code != 0:
                print(f"      重放失败（退出码 {code}）：{message}")
                continue

            if args.refresh:
                print(refresh_case(case, target_root))
                passed += 1
                print("      ⇒ 已刷新（下次不带 --refresh 再跑一遍做校验）")
                continue

            ok, detail = compare(case, target_root)
            print(detail)

            # 场景案例还带着仿真产物，一并重放（验证求解器确定性）
            if case["schema"] == "scene_request/1":
                poses_ok, poses_detail = replay_poses(case, target_root)
                if poses_detail:
                    print(poses_detail)
                ok = ok and poses_ok

            if ok:
                passed += 1
                print("      ⇒ 逐字节一致")
            else:
                print("      ⇒ 产物与记录不符：输入卡漏记了某个自由度")
    finally:
        if args.keep:
            print()
            print(f"临时目录已保留：{temporary}")
        else:
            shutil.rmtree(temporary, ignore_errors=True)

    print()
    print(_LINE)
    print(f"结果：{passed}/{len(cases)} 个案例可逐字节复现")
    print(_LINE)
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
