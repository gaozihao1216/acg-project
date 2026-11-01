# -*- coding: utf-8 -*-
"""文档数字的守卫测试：表格必须与真值逐字一致。

真值分两类，都**不在文档里**：

- 能力矩阵 → ``src/physics/capabilities.py``（另有 ``test_capabilities.py`` 专门守）
- 基准数字 / 产物字节数 → ``results/`` 里的存档与目录

这里守的是"文档只是派生物"这一条：改了存档、重跑了基准、产物重新生成，而文档没
同步，测试立刻红。手改文档同样会红——它会在下一次 ``sync_docs.py`` 被覆盖。

运行::

    python Code/tests/test_evidence.py
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src import evidence

_OPEN_PATTERN = re.compile(r"<!--\s*(?:evidence:[A-Za-z0-9_]+|capabilities):begin\s*-->")
_CLOSE_PATTERN = re.compile(r"<!--\s*(?:evidence:[A-Za-z0-9_]+|capabilities):end\s*-->")


def _markdown_files() -> list[str]:
    """仓库里所有 .md。"""
    return evidence.iter_markdown(str(evidence.PROJECT_ROOT))


def test_renderers_work() -> tuple[bool, str]:
    """每个登记的渲染器都能产出非空表格。

    渲染失败通常意味着**存档格式变了**（或少了一行），而不是文档写错。
    这时正确做法是修解析器 / 重跑基准，绝不是手改文档。
    """
    problems = []
    notes = []
    for name in sorted(evidence.RENDERERS):
        try:
            text = evidence.render(name)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}：{type(exc).__name__}: {exc}")
            continue
        if not text.strip():
            problems.append(f"{name}：渲染结果为空")
            continue
        if "|" not in text:
            problems.append(f"{name}：渲染结果不含表格")
            continue
        notes.append(f"{name}({len(text.splitlines())} 行)")

    return not problems, ("；".join(problems) if problems else "，".join(notes))


def test_archive_rows_plausible() -> tuple[bool, str]:
    """解析出的行数不能塌陷。

    空表会在 :func:`test_renderers_work` 里被抓到，但"只吃到一两行"不会——
    那正是解析器写坏时的典型症状，而表看起来仍然完整。
    """
    limits = {
        "broadphase_spread": 3,
        "broadphase_dense": 3,
        "broadphase_cell": 3,
        "solver_groove": 5,
        "solver_pyramid": 5,
        "warm_start": 8,
        "artifacts": 4,
    }
    problems = []
    notes = []
    for name, minimum in limits.items():
        rows = [
            line for line in evidence.render(name).splitlines()
            if line.startswith("| ") and "---" not in line
        ]
        data_rows = max(0, len(rows) - 1)  # 去掉表头
        notes.append(f"{name}={data_rows}")
        if data_rows < minimum:
            problems.append(f"{name} 只有 {data_rows} 行（至少应有 {minimum} 行）")

    return not problems, ("；".join(problems) if problems else "，".join(notes))


def test_markers_balanced() -> tuple[bool, str]:
    """生成块的起止标记必须成对。

    ``find_blocks`` 只匹配**成对**的标记，所以有人删掉一个 end 标记时，那块会被
    静静地忽略、文档继续显示旧表格，而测试全绿——这是这个机制最危险的失效方式。
    所以这里直接数标记。
    """
    problems = []
    total = 0
    for path in _markdown_files():
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        opens = len(_OPEN_PATTERN.findall(text))
        closes = len(_CLOSE_PATTERN.findall(text))
        found = len(evidence.find_blocks(text))
        if opens != closes:
            problems.append(
                f"{os.path.relpath(path, evidence.PROJECT_ROOT)}：begin {opens} 个 / "
                f"end {closes} 个"
            )
        elif found != opens:
            problems.append(
                f"{os.path.relpath(path, evidence.PROJECT_ROOT)}：{opens} 个 begin 但只"
                f"配对出 {found} 块（嵌套或跨块？）"
            )
        total += opens

    return not problems, ("；".join(problems) if problems else f"{total} 处标记成对")


def test_docs_blocks_current() -> tuple[bool, str]:
    """文档里每块的正文必须与当前渲染结果逐字一致。"""
    problems = []
    seen: dict[str, str] = {}

    for path in _markdown_files():
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        relative = os.path.relpath(path, evidence.PROJECT_ROOT)
        for name, content, _start, _end in evidence.find_blocks(text):
            expected = evidence.render(name).strip()
            seen.setdefault(name, relative)
            if content.strip() != expected:
                problems.append(f"{relative} [{name}]：{_first_difference(content.strip(), expected)}")

    missing = sorted(set(evidence.RENDERERS) - set(seen))
    if missing:
        problems.append(f"这些渲染器没有任何文档引用：{missing}（标记被删了？）")

    if not seen:
        return False, "文档里找不到任何生成块，约束已失效"

    return not problems, ("；".join(problems) if problems else
                          f"{len(seen)} 类块全部与真值一致：" + "，".join(sorted(seen)))


def _first_difference(actual: str, expected: str) -> str:
    """指出第一处不同的行。"""
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    for index in range(max(len(actual_lines), len(expected_lines))):
        got = actual_lines[index] if index < len(actual_lines) else "<缺失>"
        want = expected_lines[index] if index < len(expected_lines) else "<多余>"
        if got != want:
            return f"第 {index + 1} 行 文档={got!r} 应为={want!r}"
    return "内容相同但首尾空白不同"


CHECKS = [
    ("1. 所有渲染器可用", test_renderers_work),
    ("2. 存档解析出的行数没有塌陷", test_archive_rows_plausible),
    ("3. 生成块标记成对", test_markers_balanced),
    ("4. 文档数字与真值逐字一致", test_docs_blocks_current),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("文档数字（基准 / 产物字节数 / 能力矩阵）")
    print("=" * 88)

    passed = 0
    for title, func in CHECKS:
        try:
            ok, message = func()
        except Exception as exc:  # noqa: BLE001
            ok, message = False, f"抛出异常：{type(exc).__name__}: {exc}"

        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {message}")
        if ok:
            passed += 1

    print("-" * 88)
    print(f"结果：{passed}/{len(CHECKS)} 通过")
    print("=" * 88)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
