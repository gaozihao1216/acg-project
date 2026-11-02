# -*- coding: utf-8 -*-
"""文档里的"数字"只从存档生成，不手抄。

为什么需要它
------------
能力矩阵（``capabilities.py``）解决的是"支不支持"这一类事实。另有两类事实同样
会被抄错，而且抄错之后**没人能发现**：

1. **性能基准数字**：``docs/08`` 的加速比表、``docs/05`` 的求解器策略表。
   它们的真值在 ``results/w2/bench_*.txt``——基准脚本写出来的存档。
   实测对账：``docs/08`` 的 spread N=400 写的是 45.33×，存档里是 49.45×。
   不是谁算错，是文档抄的是另一次运行的输出。
2. **产物字节数**：``docs/13`` 用它当"可逐字节复现"的证据，而文件一改数字就过期
   （本轮就抓到 ``stack.poses.json`` 32911 → 32969）。

做法与能力矩阵相同：真值只有一处（存档文件 / 目录扫描），文档里的表格由
:func:`render` **生成**，``tests/test_evidence.py`` 逐块比对。手改文档会在下一次
``sync_docs.py`` 被覆盖，并立刻让测试变红。

标记格式::

    <!-- evidence:<名字>:begin -->
    ...生成内容...
    <!-- evidence:<名字>:end -->

名字 → 渲染函数的登记在 :data:`RENDERERS`。新增一类事实只需加一个渲染函数。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .physics import capabilities

#: 仓库根目录（本文件在 ``Code/src/``，上两级即仓库根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 基准存档所在目录
BENCH_DIR = PROJECT_ROOT / "results" / "w2"

#: 产物案例目录
CASE_ROOT = PROJECT_ROOT / "results" / "w3"

_BENCH_BROADPHASE = BENCH_DIR / "bench_broadphase.txt"
_BENCH_SOLVER = BENCH_DIR / "bench_solver_strategy.txt"
_BENCH_WARM = PROJECT_ROOT / "results" / "w4" / "bench_warm_start.txt"


# ---------------------------------------------------------------------------
# 标记与块操作
# ---------------------------------------------------------------------------

def begin_marker(name: str) -> str:
    """生成块的起始标记。"""
    return f"<!-- evidence:{name}:begin -->"


def end_marker(name: str) -> str:
    """生成块的结束标记。"""
    return f"<!-- evidence:{name}:end -->"


#: 匹配任意生成块（含能力矩阵那块，它由 :func:`render` 统一处理）
_BLOCK_PATTERN = re.compile(
    r"<!--\s*(?:evidence:(?P<evidence>[A-Za-z0-9_]+)|capabilities):begin\s*-->"
    r"(?P<content>.*?)"
    r"<!--\s*(?:evidence:(?P=evidence)|capabilities):end\s*-->",
    re.DOTALL,
)


def find_blocks(text: str) -> list[tuple[str, str, int, int]]:
    """找出文本里所有生成块。

    返回 ``[(名字, 内容, 内容起止下标), ...]``，按出现顺序。名字为 ``capabilities``
    的那块也在这里出现——它由 :func:`render` 分派到能力矩阵渲染器，于是同步脚本
    只需要一套逻辑。
    """
    blocks = []
    for match in _BLOCK_PATTERN.finditer(text):
        name = match.group("evidence") or "capabilities"
        blocks.append((name, match.group("content"), match.start("content"), match.end("content")))
    return blocks


def sync_text(text: str) -> tuple[str, int]:
    """把文本里每个生成块的内容换成最新渲染结果，返回（新文本, 块数）。

    从后往前替换，下标不会因为前面的替换而失效。
    """
    blocks = find_blocks(text)
    for name, _content, start, end in reversed(blocks):
        text = text[:start] + "\n" + render(name) + "\n" + text[end:]
    return text, len(blocks)


# ---------------------------------------------------------------------------
# 渲染器：真值 → Markdown
# ---------------------------------------------------------------------------

def render(name: str) -> str:
    """按名字渲染；未登记的名字直接报错（静默返回空会让文档被清空）。"""
    try:
        renderer = RENDERERS[name]
    except KeyError as exc:
        raise ValueError(
            f"未登记的生成块：{name!r}；可用：{sorted(RENDERERS)}"
        ) from exc
    return renderer()


def _read_text(path: Path) -> str:
    """读存档；UTF-8 优先，退到 GBK。

    基准脚本在不同终端下写过不同编码，这里宽容一点——但**内容**不做任何猜测，
    解析不出来就报错，不会静默跳过行。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到存档 {path}。先跑对应的基准脚本再同步文档。"
        )
    raw = path.read_bytes()
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} 既不是 UTF-8 也不是 GBK")


def _pretty(token: str) -> str:
    """把存档里的裸 token 变成表格里的写法。

    ``127.50us`` → ``127.50 µs``、``1.76x`` → ``1.76×``、``19900`` → ``19 900``。
    千位分隔用空格而不是逗号，避免和 markdown 表格的分隔符混在一起。
    """
    if token.isdigit():
        return f"{int(token):,}".replace(",", " ") if len(token) >= 4 else token

    for suffix, replacement in (("us", "µs"), ("ms", "ms"), ("x", "×")):
        if not token.endswith(suffix):
            continue
        body = token[: -len(suffix)]
        try:
            value = float(body)
        except ValueError:
            continue
        if suffix == "x":
            # 倍数列：190.0 → 190、79800.0 → 79 800，与其余列的分隔风格一致
            if value >= 1000:
                return f"{int(round(value)):,}".replace(",", " ") + "×"
            return f"{value:.2f}".rstrip("0").rstrip(".") + "×"
        return f"{body} {replacement}"
    return token


def _markdown_table(
    header: list[str],
    rows: list[list[str]],
    left: tuple[int, ...] = (0,),
) -> str:
    """拼一张 markdown 表：数字列右对齐便于对照存档，文本列左对齐。

    ``left`` 给出需要左对齐的列号（默认只有第 0 列）。
    """
    aligns = ["---" if index in left else "---:" for index in range(len(header))]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _broadphase_sections() -> dict[str, list[list[str]]]:
    """把 ``bench_broadphase.txt`` 解析成三个小节的行数据。

    解析规则很朴素：按小节标题切换，数据行按列数识别（10 列 = N 扫描表，
    7 列 = 格子边长表）。识别不出任何一行直接报错——**空表比错的表更危险**。
    """
    section = None
    spread: list[list[str]] = []
    dense: list[list[str]] = []
    cell: list[list[str]] = []

    for line in _read_text(_BENCH_BROADPHASE).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "布局" in stripped and "spread" in stripped:
            section = "spread"
            continue
        if "布局" in stripped and "dense" in stripped:
            section = "dense"
            continue
        if "格子边长敏感性" in stripped:
            section = "cell"
            continue

        tokens = stripped.split()
        if section in ("spread", "dense") and len(tokens) == 10 and tokens[0].isdigit():
            (spread if section == "spread" else dense).append(
                [tokens[0]] + [_pretty(token) for token in tokens[1:]]
            )
        elif section == "cell" and len(tokens) == 7 and re.fullmatch(r"\d+\.\d+m", tokens[0]):
            cell.append([_pretty(token) for token in tokens])

    if not (spread and dense and cell):
        raise ValueError(
            f"{_BENCH_BROADPHASE} 解析失败（spread={len(spread)} dense={len(dense)} "
            f"cell={len(cell)} 行）。存档格式变了？先修解析再同步文档。"
        )

    return {"spread": spread, "dense": dense, "cell": cell}


_PAIR_HEADER = [
    "N", "检测 暴力", "检测 哈希", "检测加速", "候选对 暴力", "候选对 哈希",
    "窄相减少", "整步 暴力", "整步 哈希", "端到端加速",
]

_CELL_HEADER = ["格子边长", "相对默认", "检测耗时", "格子数", "兜底物体", "候选对", "相对默认"]

_BROADPHASE_SOURCE = "*来源：`results/w2/bench_broadphase.txt`*\n\n"


def render_broadphase_spread() -> str:
    """spread 布局的加速比表。"""
    return _BROADPHASE_SOURCE + _markdown_table(
        _PAIR_HEADER, _broadphase_sections()["spread"]
    )


def render_broadphase_dense() -> str:
    """dense 布局的加速比表。"""
    return _BROADPHASE_SOURCE + _markdown_table(
        _PAIR_HEADER, _broadphase_sections()["dense"]
    )


def render_broadphase_cell() -> str:
    """格子边长敏感性表。"""
    return _BROADPHASE_SOURCE + _markdown_table(
        _CELL_HEADER, _broadphase_sections()["cell"]
    )


def _solver_sections() -> dict[str, list[list[str]]]:
    """把 ``bench_solver_strategy.txt`` 解析成两个场景的行数据。"""
    section = None
    groove: list[list[str]] = []
    pyramid: list[list[str]] = []

    for line in _read_text(_BENCH_SOLVER).splitlines():
        stripped = line.strip()
        if "场景一" in stripped:
            section = "groove"
            continue
        if "场景二" in stripped:
            section = "pyramid"
            continue
        if stripped.startswith("===") or stripped.startswith("---"):
            continue

        tokens = stripped.split()
        try:
            numeric = all(re.fullmatch(r"-?\d+(?:\.\d+)?", token) for token in tokens)
        except re.error:  # pragma: no cover - 正则本身编译失败才会到这里
            numeric = False
        if not numeric:
            continue

        if section == "groove" and len(tokens) == 8:
            groove.append(tokens)
        elif section == "pyramid" and len(tokens) == 6:
            pyramid.append(tokens)

    if not (groove and pyramid):
        raise ValueError(
            f"{_BENCH_SOLVER} 解析失败（场景一={len(groove)} 场景二={len(pyramid)} 行）"
        )

    return {"groove": groove, "pyramid": pyramid}


_GROOVE_HEADER = [
    "substeps", "iters", "遍历", "平均下沉 mm", "最差 mm", "侧移 mm", "余速", "ms/步",
]
_PYRAMID_HEADER = ["substeps", "iters", "遍历", "偏离参照 mm", "余速", "ms/步"]

_SOLVER_SOURCE = "*来源：`results/w2/bench_solver_strategy.txt`*\n\n"


def render_solver_groove() -> str:
    """场景一（凹槽静置）的策略表。"""
    return _SOLVER_SOURCE + _markdown_table(
        _GROOVE_HEADER, _solver_sections()["groove"]
    )


def render_solver_pyramid() -> str:
    """场景二（金字塔多接触链）的策略表。"""
    return _SOLVER_SOURCE + _markdown_table(
        _PYRAMID_HEADER, _solver_sections()["pyramid"]
    )


def _warm_sections() -> dict[str, list[list[str]]]:
    """把 ``bench_warm_start.txt`` 解析成"场景 → 行"。

    存档格式（由 ``scripts/bench_warm_start.py`` 写出）::

        == 6 层盒塔 ==
        warm   iters  峰值穿透mm  末段穿透mm  塔顶漂移mm  最大速度m/s  ms/步
        off        1     5.264     4.581     ...
    """
    sections: dict[str, list[list[str]]] = {}
    current: str | None = None

    for line in _read_text(_BENCH_WARM).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("==") and stripped.endswith("=="):
            current = stripped.strip("= ").strip()
            sections[current] = []
            continue
        if stripped.startswith("warm") or current is None:
            continue

        tokens = stripped.split()
        if len(tokens) == 7 and tokens[0] in ("on", "off"):
            sections[current].append(tokens)

    if not sections or not all(sections.values()):
        raise ValueError(
            f"{_BENCH_WARM} 解析失败（小节：{ {k: len(v) for k, v in sections.items()} }）。"
            f"先跑 Code/scripts/bench_warm_start.py 重新生成存档。"
        )
    return sections


_WARM_HEADER = [
    "暖启动", "迭代", "峰值穿透 mm", "末段穿透 mm", "顶层漂移 mm", "最大速度 m/s", "ms/步",
]

_WARM_SOURCE = "*来源：`results/w4/bench_warm_start.txt`*\n\n"


def render_warm_start() -> str:
    """暖启动 A/B 对照表（每个场景一张）。"""
    tables = []
    for title, rows in _warm_sections().items():
        formatted = [
            ["开" if row[0] == "on" else "关", row[1], row[2], row[3], row[4], row[5], row[6]]
            for row in rows
        ]
        tables.append(f"**{title}**\n\n" + _markdown_table(_WARM_HEADER, formatted))
    return _WARM_SOURCE + "\n\n".join(tables)


#: 视为"可复现证据"的产物后缀。``*.expanded.scene.json`` 是渲染前的派生物，
#: 每次重新生成、不进证据链，所以排除。
_EVIDENCE_SUFFIXES = (".asset.json", ".scene.json", ".engine.json", ".poses.json")


def _normalized_size(path: Path) -> int:
    """按 LF 归一化后的字节数。

    这些 JSON 产物同时被当作"可逐字节复现"的证据，但 ``st_size`` 会随检出时的
    换行转换而变：``core.autocrlf=true`` 的机器上 ``cloud.asset.json`` 是 5057
    字节，LF 机器上是 4854——差值 203 正好等于行数。同一个提交在两台机器上得出
    两个"真值"，这条证据就失效了。

    文档要记的是**内容**的字节数，不是某台机器磁盘分配的字节数，所以先统一换行
    再计数。仓库另外用 ``.gitattributes`` 让 ``*.json`` 一律以 LF 检出。
    """
    raw = path.read_bytes()
    return len(raw.replace(b"\r\n", b"\n"))


def render_artifacts() -> str:
    """``docs/13`` 的产物字节数表，**现场扫描目录**得到。

    字节数是最廉价、也最不容辩解的证据：文件被重跑一次而内容变了，数字就变。
    注意这里只给"多大"，不给"一致"——逐字节复现要跑 ``reproduce.py``（会重放
    仿真，较重），所以那句话留在正文，由人跑。

    计数口径见 :func:`_normalized_size`：LF 归一化，避免换行转换让数字漂移。
    """
    if not CASE_ROOT.is_dir():
        raise FileNotFoundError(f"找不到案例目录 {CASE_ROOT}")

    rows: list[list[str]] = []
    for case_dir in sorted(p for p in CASE_ROOT.iterdir() if p.is_dir()):
        output = case_dir / "output"
        if not output.is_dir():
            continue
        for artifact in sorted(output.iterdir()):
            name = artifact.name
            if not name.endswith(_EVIDENCE_SUFFIXES) or ".expanded." in name:
                continue
            size = _normalized_size(artifact)
            rows.append([
                case_dir.name,
                f"`{name}`",
                f"{size:,}".replace(",", " "),
            ])

    if not rows:
        raise ValueError(f"{CASE_ROOT} 下没有任何可复现产物，证据表会是空的")

    return (
        "*来源：现场扫描 `results/w3/*/output/`*\n\n"
        + _markdown_table(["案例", "产物", "字节数"], rows, left=(0, 1))
    )


#: 名字 → 渲染函数。新增一类事实只需在这里登记一个函数。
RENDERERS = {
    "capabilities": capabilities.markdown_table,
    "broadphase_spread": render_broadphase_spread,
    "broadphase_dense": render_broadphase_dense,
    "broadphase_cell": render_broadphase_cell,
    "solver_groove": render_solver_groove,
    "solver_pyramid": render_solver_pyramid,
    "warm_start": render_warm_start,
    "artifacts": render_artifacts,
}


# ---------------------------------------------------------------------------
# 文档遍历
# ---------------------------------------------------------------------------

#: 遍历时跳过的目录（版本库、缓存、虚拟环境）
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def iter_markdown(root: str | os.PathLike[str]) -> list[str]:
    """列出仓库里所有 ``.md``，顺序稳定（便于输出可复现）。"""
    found = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".md"):
                found.append(os.path.join(current, name))
    return found


if __name__ == "__main__":  # pragma: no cover - 手工查看用
    for block_name in sorted(RENDERERS):
        print(f"===== {block_name} =====")
        print(render(block_name))
        print()
