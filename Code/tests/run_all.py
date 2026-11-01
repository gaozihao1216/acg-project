# -*- coding: utf-8 -*-
"""一键运行全部物理测试。

用法::

    python Code/tests/run_all.py

每个测试模块自己会打印详细结果，这里只负责汇总通过与否，
并返回一个可供 CI / 批处理判断的退出码。
"""

from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: 按顺序执行的测试脚本（文件名，不含扩展名）
TEST_MODULES = [
    "test_rigid_sphere",
    "test_offcenter_collision",
    "test_multicontact",
    "test_toi_split",
    "test_toi_convex",
    "test_box_collision",
    "test_box_box",
    "test_broadphase",
    "test_massprops",
    "test_hull",
    "test_curved",
    "test_compound",
    "test_object_input",
    "test_asset",
    "test_capabilities",
    "test_evidence",
    "test_warm_start",
]


def main() -> int:
    """逐个执行测试脚本，汇总结果。"""
    print(f"共 {len(TEST_MODULES)} 个测试模块\n")

    failures = []
    for module in TEST_MODULES:
        script = os.path.join(_HERE, f"{module}.py")
        if not os.path.isfile(script):
            print(f"[MISSING] {module}.py")
            failures.append(module)
            continue

        # 每个模块独立进程运行：互不污染全局状态，也能单独定位崩溃
        completed = subprocess.run([sys.executable, script], cwd=os.path.dirname(_HERE))
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"\n>>> {module}: {status}\n")
        if completed.returncode != 0:
            failures.append(module)

    print("=" * 60)
    if failures:
        print(f"失败模块：{', '.join(failures)}")
        return 1
    print(f"全部 {len(TEST_MODULES)} 个测试模块通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
