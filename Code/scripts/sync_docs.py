# -*- coding: utf-8 -*-
"""把所有文档里的"生成块"重写成最新真值。

覆盖两类事实：

- ``capabilities``：窄相 / TOI 的覆盖范围（真值在 ``src/physics/capabilities.py``）
- ``broadphase`` / ``solver_strategy`` / ``artifacts``：基准数字与产物字节数
  （真值在 ``results/`` 里的存档与目录）

用法::

    python Code/scripts/sync_docs.py           # 就地重写
    python Code/scripts/sync_docs.py --check    # 只报告差异，不写文件

正确的工作流是"改真值 → 同步 → 跑测试"，而不是直接改文档里的数字：
手改会在下次同步时被覆盖，而 ``tests/test_evidence.py`` 会先红。
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src import evidence


def main(argv: list[str] | None = None) -> int:
    """同步（或检查）所有文档块。"""
    parser = argparse.ArgumentParser(description="同步文档里的生成块")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查是否已同步；有差异时返回码为 1",
    )
    parser.add_argument(
        "--root",
        default=str(evidence.PROJECT_ROOT),
        help="仓库根目录（默认按脚本位置推断）",
    )
    args = parser.parse_args(argv)

    stale = []
    total_blocks = 0

    for path in evidence.iter_markdown(args.root):
        with open(path, "r", encoding="utf-8") as handle:
            original = handle.read()
        if "<!--" not in original:
            continue

        try:
            updated, blocks = evidence.sync_text(original)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[跳过] {os.path.relpath(path, args.root)}：{exc}")
            return 1

        total_blocks += blocks
        if updated != original:
            stale.append(os.path.relpath(path, args.root))
            if not args.check:
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(updated)

    if total_blocks == 0:
        print("没有找到任何生成块——文档失去了约束，先确认标记还在。")
        return 1

    if args.check:
        if stale:
            print(f"以下 {len(stale)} 个文件的生成块已过期：")
            for relative in stale:
                print(f"  - {relative}")
            print("运行 python Code/scripts/sync_docs.py 重写。")
            return 1
        print(f"{total_blocks} 处生成块已是最新。")
        return 0

    if stale:
        print(f"已更新 {len(stale)} 个文件（共 {total_blocks} 处块）：")
        for relative in stale:
            print(f"  - {relative}")
    else:
        print(f"{total_blocks} 处生成块已是最新，未改动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
