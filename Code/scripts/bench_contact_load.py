"""Measure the current solver under real contacts without changing physics settings.

Run from project root: python Code/scripts/bench_contact_load.py
Raw wall timings, exclusive stage timings and cProfile are separate passes.
Scene loading, warm-up, deepcopy, diagnostics and serialization are not timed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import cProfile
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import pickle
import pstats
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from src.geom.geometry import BoxGeometry
from src.geom.inertia import compute_inertia
from src.physics.body import RigidBody
from src.physics.broadphase import SpatialHashBroadphase, BruteForceBroadphase
from src.physics.contact_cache import ContactManager
from src.physics.narrowphase import GroundPlane
from src.physics.solver import SequentialImpulseSolver
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec, load_scene
import src.physics.world as world_module


def state_hash(world):
    state = [(b.name, b.position, b.orientation, b.linear_velocity, b.angular_momentum)
             for b in world.bodies]
    cache = [(key, e.anchor_a, e.anchor_b, e.normal, e.normal_impulse, e.h,
              e.seen_epoch, e.warm_epoch) for key, e in sorted(world.solver.contacts.entries.items())]
    payload = [state, cache, world.time, world.step_count, world.solver.contacts.epoch]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build(case, count):
    if case == "curved_demo":
        scene = load_scene(str(ROOT / "Code/scenes/demo_curved.json"))
        world = PhysicsWorld.from_scene(scene)
        for _ in range(round(.75 / world.simulation.physics_dt)):
            world.step()
        return world
    geometry = BoxGeometry((.5, .5, .5))
    inertia = compute_inertia(geometry, 1)
    bodies = []
    side = math.ceil(math.sqrt(math.ceil(count / 5))) if case == "stack" else math.ceil(math.sqrt(count))
    for i in range(count):
        if case == "stack":
            column, layer = divmod(i, 5)
            position = ((column % side) * .65, (column // side) * .65, .2499 + layer * .4999)
        else:
            position = ((i % side) * 1.5, (i // side) * 1.5, .2505 if case == "impact" else 10)
        body = RigidBody(str(i), geometry, position=position, mass=1, inertia_local=inertia,
                         restitution=.1, friction=.5)
        if case == "impact":
            body.linear_velocity = (0, 0, -2)
        bodies.append(body)
    world = PhysicsWorld(bodies, GroundPlane(z=0, friction=.55, restitution=.12),
                         SimulationSpec(enabled=True))
    # Persistent support and free-flight cases start from a warmed, deterministic state.
    if case != "impact":
        for _ in range(4):
            world.step()
    return world


class Stages:
    """Exclusive wall times: nested measured calls are subtracted from parents."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.counts = defaultdict(int)
        self.stack = []
        self.restore = []

    def wrap(self, target, name, stage, count=None):
        original = getattr(target, name)

        def measured(*args, **kwargs):
            label = stage(self.stack) if callable(stage) else stage
            frame = [time.perf_counter(), 0.0, label]
            self.stack.append(frame)
            result = None
            try:
                result = original(*args, **kwargs)
                if count is not None:
                    count(result, args)
                return result
            finally:
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                self.seconds[label] += elapsed - frame[1]
                self.calls[label] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed

        self.restore.append((target, name, original))
        setattr(target, name, measured)

    def __enter__(self):
        def pairs(result, _args):
            self.counts["candidate_pairs_returned"] += len(result)

        def contacts(result, _args):
            self.counts["manifolds_returned"] += len(result)
            self.counts["points_returned"] += sum(len(m.points) for m in result)

        def velocity(result, _args):
            self.counts["velocity_constraints"] += result

        def position(result, _args):
            self.counts["active_position_constraints"] += result

        def substep(_result, args):
            self.counts["actual_toi_events"] += args[0].last_event_count

        self.wrap(PhysicsWorld, "substep", "world_other", substep)
        self.wrap(SpatialHashBroadphase, "candidate_pairs", "broadphase", pairs)
        self.wrap(BruteForceBroadphase, "candidate_pairs", "broadphase", pairs)
        self.wrap(world_module, "detect_contacts", "narrowphase", contacts)
        self.wrap(world_module, "earliest_toi_event", "toi_query")
        self.wrap(SequentialImpulseSolver, "solve", "velocity_solve", velocity)
        self.wrap(SequentialImpulseSolver, "solve_positions", "position_solve", position)
        self.wrap(SequentialImpulseSolver, "_build_constraints",
                  lambda stack: "position_prepare" if stack and stack[-1][2] == "position_solve" else "velocity_prepare")
        self.wrap(PhysicsWorld, "_integrate_velocities", "external_forces")
        self.wrap(PhysicsWorld, "_integrate_positions", "pose_integration")
        return self

    def __exit__(self, *_):
        for target, name, original in reversed(self.restore):
            setattr(target, name, original)


def profile_world(base, steps, directory, key):
    world = copy.deepcopy(base)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    for _ in range(steps):
        world.step()
    profiler.disable()
    elapsed = time.perf_counter() - start
    profiler.dump_stats(str(directory / f"{key}.prof"))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative").print_stats(35)
    stats.sort_stats("tottime").print_stats(35)
    (directory / f"{key}.profile.txt").write_text(stream.getvalue(), encoding="utf-8")
    entries = []
    for (filename, line, function), (primitive, calls, own, cumulative, _callers) in stats.stats.items():
        try:
            filename = str(Path(filename).relative_to(ROOT))
        except ValueError:
            pass
        entries.append(dict(file=filename, line=line, function=function, calls=calls,
                            primitive_calls=primitive, self_seconds=own, cumulative_seconds=cumulative))
    entries.sort(key=lambda e: e["cumulative_seconds"], reverse=True)
    return dict(wall_seconds=elapsed, steps=steps, functions=entries), state_hash(world)


def measure(case, count, trials, directory, profile, inputs=None, write_inputs=False):
    key = f"{case}_{count}"
    start = time.perf_counter()
    snapshot = None if inputs is None else inputs / f"{key}.pickle"
    if snapshot is not None and not write_inputs:
        # These snapshots are generated locally by this script at the baseline stage.
        base = pickle.loads(snapshot.read_bytes())
    else:
        base = build(case, count)
        if snapshot is not None:
            if snapshot.exists():
                raise FileExistsError(f"Refusing to overwrite frozen input: {snapshot}")
            snapshot.write_bytes(pickle.dumps(base, protocol=pickle.HIGHEST_PROTOCOL))
    setup_seconds = time.perf_counter() - start
    actual_count = len(base.bodies)
    steps = 1 if case == "impact" else 2 if count >= 500 else 6
    samples = []
    trial_hashes = []
    final = None
    for _ in range(trials):
        world = copy.deepcopy(base)
        for _ in range(steps):
            start = time.perf_counter()
            world.step()
            samples.append(time.perf_counter() - start)
        world.check_finite()
        trial_hashes.append(state_hash(world))
        final = world
    if len(set(trial_hashes)) != 1:
        raise AssertionError(f"{key}: repeated trials are not deterministic")

    # A separate instrumented pass avoids charging wrapper overhead to raw timings.
    instrumented = copy.deepcopy(base)
    with Stages() as stages:
        start = time.perf_counter()
        for _ in range(steps):
            instrumented.step()
        stage_wall = time.perf_counter() - start
    if state_hash(instrumented) != trial_hashes[0]:
        raise AssertionError(f"{key}: stage instrumentation changed physics")
    measured = sum(stages.seconds.values())
    stage_seconds = dict(stages.seconds)
    stage_seconds["outside_substep"] = max(0, stage_wall - measured)
    result = dict(
        case=case, bodies=actual_count, trials=trials, steps_per_trial=steps,
        simulation=base.simulation.to_dict(), setup_seconds=setup_seconds,
        initial_sim_time=base.time, measured_sim_time_per_trial=steps * base.simulation.physics_dt,
        raw_step_ms=[s * 1000 for s in samples], median_step_ms=statistics.median(samples) * 1000,
        min_step_ms=min(samples) * 1000, max_step_ms=max(samples) * 1000,
        wall_seconds_per_sim_second=statistics.median(samples) / base.simulation.physics_dt,
        stage_wall_seconds=stage_wall, exclusive_stage_seconds=stage_seconds,
        instrumentation_wall_ratio=stage_wall / (sum(samples) / trials),
        exclusive_stage_percent={k: v / stage_wall * 100 for k, v in stage_seconds.items()},
        stage_calls=dict(stages.calls), workload_counts=dict(stages.counts),
        final_max_speed=final.max_speed(), final_penetration_mm=final.max_penetration() * 1000,
        cache_entries=len(final.solver.contacts.entries), deterministic=True,
        initial_state_sha256=state_hash(base), final_state_sha256=trial_hashes[0],
        input_snapshot_sha256=None if snapshot is None else hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        final_states=[dict(name=b.name, position=b.position, orientation=b.orientation,
                           linear_velocity=b.linear_velocity, angular_momentum=b.angular_momentum)
                      for b in final.bodies],
    )
    if profile:
        result["profile"], profile_hash = profile_world(base, steps, directory, key)
        if profile_hash != trial_hashes[0]:
            raise AssertionError(f"{key}: cProfile changed physics")
    return result


def source_metadata():
    sources = {}
    paths = sorted((ROOT / "Code/src").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    aggregate = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return dict(
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        processor=platform.processor(), logical_cpus=os.cpu_count(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8"),
        source_sha256=aggregate, source_files=sources,
    )


def write_outputs(directory, payload):
    (directory / "measurements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["case", "bodies", "trials", "steps_per_trial", "median_step_ms", "min_step_ms", "max_step_ms",
              "wall_seconds_per_sim_second", "final_max_speed", "final_penetration_mm", "cache_entries"]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["results"])
    lines = ["Current contact-load measurements", "",
             "Raw median timings are uninstrumented. Stage shares come from a separate instrumented pass.",
             "Stage shares are exclusive; cProfile cumulative times overlap and must not be added.",
             "Setup/warmup, cloning, diagnostics, I/O and rendering are excluded from measured steps.",
             "No timing from historical results is reused. Source hashes identify the dirty working tree.",
             "Stacks are short warmed support loads, not a long-term settled-pile benchmark.",
             "Impact means equal-height boxes falling onto the floor synchronously.", ""]
    for r in payload["results"]:
        lines.append(f"{r['case']:12} N={r['bodies']:4} median={r['median_step_ms']:.3f} ms/step "
                     f"range=[{r['min_step_ms']:.3f},{r['max_step_ms']:.3f}] "
                     f"wall/sim={r['wall_seconds_per_sim_second']:.2f}")
        lines.append("  stages: " + ", ".join(f"{k}={v:.1f}%" for k, v in sorted(
            r["exclusive_stage_percent"].items(), key=lambda kv: -kv[1]) if v >= .1))
        lines.append("  counts (instrumented pass): " + json.dumps(r["workload_counts"], sort_keys=True))
    (directory / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 125, 500])
    parser.add_argument("--cases", nargs="+", choices=["air", "impact", "stack", "curved_demo"],
                        default=["air", "impact", "stack"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile-size", type=int, default=125)
    parser.add_argument("--include-curved", action="store_true")
    parser.add_argument("--out", default="results/performance/2026-11-27")
    parser.add_argument("--inputs", help="Frozen baseline-world directory for cross-version comparisons")
    parser.add_argument("--write-inputs", action="store_true", help="Create immutable baseline input snapshots")
    args = parser.parse_args()
    if args.trials < 1 or any(n < 1 for n in args.sizes):
        parser.error("sizes and trials must be positive")
    directory = ROOT / args.out
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "measurements.json").exists():
        parser.error("Output already contains measurements; select a fresh --out directory")
    inputs = None if args.inputs is None else ROOT / args.inputs
    if args.write_inputs and inputs is None:
        parser.error("--write-inputs requires --inputs")
    if inputs is not None:
        inputs.mkdir(parents=True, exist_ok=True)
    payload = dict(method="contact_load/1", environment=source_metadata(), arguments=vars(args), results=[])
    schedule = [(case, n) for n in args.sizes for case in args.cases if case != "curved_demo"]
    if args.include_curved or "curved_demo" in args.cases:
        schedule.append(("curved_demo", 5))
    for case, count in schedule:
        result = measure(case, count, args.trials, directory, count == args.profile_size or case == "curved_demo",
                         inputs, args.write_inputs)
        payload["results"].append(result)
        write_outputs(directory, payload)
        print(f"{case} N={result['bodies']} median={result['median_step_ms']:.3f}ms "
              f"wall/sim={result['wall_seconds_per_sim_second']:.2f} "
              f"deterministic={result['deterministic']}", flush=True)


if __name__ == "__main__":
    main()
