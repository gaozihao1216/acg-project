"""Archive exact optimization checkpoints and interleave frozen-version timings."""
import argparse
import copy
import csv
import gc
import hashlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import pickle
import platform
import statistics
import sys
import time
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/performance/optimization_2026-11-27"
STAGES = ["01_baseline", "02_position_prepare", "03_inverse_inertia", "04_ground_culling", "05_support_transform"]


def archive(name):
    directory = BASE / name
    data = json.loads((directory / "measurements.json").read_text(encoding="utf-8"))
    for path, sha in data["environment"]["source_files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
    target = directory / "source.zip"
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("Code/src", "Code/tests", "Code/scenes"):
            for p in sorted((ROOT / folder).rglob("*")):
                if p.suffix in (".py", ".json"):
                    z.write(p, p.relative_to(ROOT))
        z.write(ROOT / "Code/scripts/bench_contact_load.py", "Code/scripts/bench_contact_load.py")
    print("Archived", name, flush=True)


class VersionUnpickler(pickle.Unpickler):
    def __init__(self, buffer, alias):
        super().__init__(buffer)
        self.alias = alias

    def find_class(self, module, name):
        if module == "src" or module.startswith("src."):
            return getattr(importlib.import_module(self.alias + module[3:]), name)
        return super().find_class(module, name)


def extract_versions():
    aliases = {}
    for i, name in enumerate(STAGES):
        digest = hashlib.sha256((BASE / name / "source.zip").read_bytes()).hexdigest()[:12]
        directory = Path(tempfile.gettempdir()) / "acg_optimization_versions" / f"{name}_{digest}"
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BASE / name / "source.zip") as z:
            z.extractall(directory)
        source = directory / "Code/src"
        alias = f"contact_version_{i}"
        spec = importlib.util.spec_from_file_location(alias, source / "__init__.py",
                                                     submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        aliases[name] = alias
    return aliases


def physical_state(world):
    return [dict(name=b.name, position=list(b.position), orientation=list(b.orientation),
                 linear_velocity=list(b.linear_velocity), angular_momentum=list(b.angular_momentum))
            for b in world.bodies]


def differences(reference, actual):
    assert len(reference) == len(actual)
    maxima = dict(position=0., orientation_radians=0., linear_velocity=0., angular_momentum=0.)
    for a, b in zip(reference, actual):
        assert a["name"] == b["name"]
        for field in ("position", "linear_velocity", "angular_momentum"):
            maxima[field] = max(maxima[field], max(abs(x - y) for x, y in zip(a[field], b[field])))
        qa, qb = a["orientation"], b["orientation"]
        chord = min(math.sqrt(sum((x - s * y)**2 for x, y in zip(qa, qb))) for s in (-1, 1))
        maxima["orientation_radians"] = max(maxima["orientation_radians"], 4 * math.asin(min(1., chord / 2)))
    return maxima


def compare_sequential():
    data = {name: json.loads((BASE / name / "measurements.json").read_text(encoding="utf-8")) for name in STAGES}
    baseline = data[STAGES[0]]["results"]
    rows = []
    for ref in baseline:
        row = dict(case=ref["case"], bodies=ref["bodies"], stages={})
        previous = None
        for name in STAGES:
            current = next(r for r in data[name]["results"] if (r["case"], r["bodies"]) == (ref["case"], ref["bodies"]))
            assert current["initial_state_sha256"] == ref["initial_state_sha256"]
            assert current["input_snapshot_sha256"] == ref["input_snapshot_sha256"]
            assert current["simulation"] == ref["simulation"]
            row["stages"][name] = dict(ms=current["median_step_ms"],
                cumulative_speedup=ref["median_step_ms"] / current["median_step_ms"],
                incremental_speedup=None if previous is None else previous / current["median_step_ms"],
                bitwise_equal=current["final_state_sha256"] == ref["final_state_sha256"],
                physical_difference=differences(ref["final_states"], current["final_states"]))
            previous = current["median_step_ms"]
        rows.append(row)
    (BASE / "sequential_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return baseline


def paired(rounds):
    cases = compare_sequential()
    aliases = extract_versions()
    results = []
    for case in cases:
        key = f"{case['case']}_{case['bodies']}"
        frozen = (BASE / "frozen_inputs" / f"{key}.pickle").read_bytes()
        assert hashlib.sha256(frozen).hexdigest() == case["input_snapshot_sha256"]
        bases = {name: VersionUnpickler(io.BytesIO(frozen), alias).load() for name, alias in aliases.items()}
        timings = {name: [] for name in STAGES}
        finals = {}
        steps = case["steps_per_trial"]
        for round_index in range(rounds):
            order = STAGES[round_index % len(STAGES):] + STAGES[:round_index % len(STAGES)]
            for name in order:
                world = copy.deepcopy(bases[name])
                gc.collect()  # outside timing, same protocol for every variant
                start = time.perf_counter()
                for _ in range(steps):
                    world.step()
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed * 1000 / steps)
                world.check_finite()
                state = physical_state(world)
                if name in finals:
                    assert finals[name] == state, (key, name, "nondeterministic")
                finals[name] = state
        row = dict(case=case["case"], bodies=case["bodies"], steps_per_round=steps, rounds=rounds, stages={})
        for i, name in enumerate(STAGES):
            row["stages"][name] = dict(
                raw_ms=timings[name], median_ms=statistics.median(timings[name]),
                cumulative_speedup=statistics.median(a / b for a, b in zip(timings[STAGES[0]], timings[name])),
                incremental_speedup=1. if not i else statistics.median(a / b for a, b in zip(timings[STAGES[i-1]], timings[name])),
                physical_difference=differences(finals[STAGES[0]], finals[name]),
            )
        results.append(row)
        (BASE / "paired_comparison.json").write_text(json.dumps(dict(
            protocol="paired/1", order="rotating variant order within each case/round", gc="collect before each untimed-to-timed boundary; enabled during timing",
            python=sys.version, platform=platform.platform(),
            comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_archives={name: hashlib.sha256((BASE/name/"source.zip").read_bytes()).hexdigest() for name in STAGES},
            stages=STAGES, results=results), indent=2), encoding="utf-8")
        print(key, " ".join(f"{name.split('_')[0]}:{row['stages'][name]['median_ms']:.2f}ms" for name in STAGES), flush=True)
    with (BASE / "paired_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "bodies", "stage", "median_ms", "incremental_speedup", "cumulative_speedup"])
        for r in results:
            for name, v in r["stages"].items():
                writer.writerow([r["case"], r["bodies"], name, v["median_ms"], v["incremental_speedup"], v["cumulative_speedup"]])


def plot_results():
    isolated_runtime = Path(tempfile.gettempdir()) / "acg_perf_plot_runtime"
    if isolated_runtime.exists():
        sys.path.insert(0, str(isolated_runtime))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads((BASE / "paired_comparison.json").read_text(encoding="utf-8"))
    rows = {(r["case"], r["bodies"]): r for r in data["results"]}
    choices = [("air", 125, "Air / 125 boxes"), ("stack", 125, "Stack / 125 boxes"),
               ("stack", 500, "Stack / 500 boxes"), ("curved_demo", 5, "Curved demo / 5 bodies")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for case, count, label in choices:
        values = rows[case, count]["stages"]
        axes[0].plot(range(5), [values[name]["cumulative_speedup"] for name in STAGES], marker="o", label=label)
    axes[0].set_xticks(range(5), ["Baseline", "1 Position\nprepare", "2 Inertia\ncache", "3 Ground\nculling", "4 Support\ntransform"])
    axes[0].set_ylabel("Cumulative speedup (paired median)")
    axes[0].axhline(1, color="gray", lw=.8, ls="--")
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for offset, name, label, color in [(-.19, STAGES[0], "Baseline", "#9ca3af"), (.19, STAGES[-1], "Final", "#2563eb")]:
        values = [rows[c, n]["stages"][name]["median_ms"] for c, n, _ in choices]
        bars = axes[1].bar([i + offset for i in range(4)], values, width=.36, label=label, color=color)
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x()+bar.get_width()/2, value*1.07, f"{value:.1f}", ha="center", fontsize=8)
    axes[1].set_xticks(range(4), ["Air 125", "Stack 125", "Stack 500", "Curved 5"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median ms / physics step (log scale)")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend()
    fig.suptitle(f"Contact-load optimization: {data['results'][0]['rounds']} interleaved rounds, identical frozen inputs")
    fig.tight_layout()
    fig.savefig(BASE / "speedup_progress.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=STAGES)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.archive:
        archive(args.archive)
    if args.paired:
        paired(args.rounds)
    if args.plot:
        plot_results()
