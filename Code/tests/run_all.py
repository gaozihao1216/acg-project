# -*- coding: utf-8 -*-
"""一键运行全部物理测试。

用法::

    python Code/tests/run_all.py

每个测试模块自己会打印详细结果，这里只负责汇总通过与否，
并返回一个可供 CI / 批处理判断的退出码。

``TEST_MODULES`` 是**手写**的清单（顺序有意义：先塞物理核心、再塞证据类），
所以它一定会落后于文件系统——新增 ``test_*.py`` 却忘了登记时，那个模块就成了
"单独跑能过、一键跑不跑"的幽灵。:func:`check_registry` 每次启动先对账，
少登记、多登记都直接失败。
"""

from __future__ import annotations

import glob
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
    "test_collision_reliability",
    "test_dynamics_contact",
    "test_box_collision",
    "test_box_box",
    "test_broadphase",
    "test_massprops",
    "test_primitive_geometry",
    "test_hull",
    "test_curved",
    "test_compound",
    "test_inertia_modes",
    "test_ground_generic",
    "test_contact_identity",
    "test_contact_region",
    "test_poly2d",
    "test_support_patch",
    "test_patch",
    "test_diagnostics",
    "test_object_input",
    "test_asset",
    "test_capabilities",
    "test_evidence",
    "test_warm_start",
]


def check_registry() -> tuple[list[str], list[str]]:
    """清单与磁盘上的 ``test_*.py`` 对账，返回 ``(漏登记, 多登记)``。

    两边都用模块名（不含扩展名）比较。``run_all.py`` 自己不是 ``test_*.py``，
    不会混进来。
    """
    registered = set(TEST_MODULES)
    existing = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(_HERE, "test_*.py"))
    }
    return sorted(existing - registered), sorted(registered - existing)


def main() -> int:
    """逐个执行测试脚本，汇总结果。"""
    missing_from_list, missing_on_disk = check_registry()
    if missing_from_list or missing_on_disk:
        print("=" * 60)
        if missing_from_list:
            print(f"这些测试文件没有登记到 TEST_MODULES：{', '.join(missing_from_list)}")
        if missing_on_disk:
            print(f"TEST_MODULES 里这些模块在磁盘上不存在：{', '.join(missing_on_disk)}")
        print("=" * 60)
        return 1

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
