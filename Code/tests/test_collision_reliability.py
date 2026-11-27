"""Regression scenes for near-contact CCD, swept candidates and event budgets."""

from __future__ import annotations

import os
import sys
import unittest
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geom.geometry import BoxGeometry, ConvexHullGeometry, CylinderGeometry, SphereGeometry
from src.geom.inertia import compute_inertia
from src.mathutils import quaternion as quat
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane
from src.physics.toi import body_toi, ground_toi
from src.physics.support import support_world
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec


def body(name, geometry, position, velocity=(0, 0, 0), static=False, restitution=0):
    result = RigidBody(
        name, geometry, position=position, orientation=geometry.inertia_frame(), mass=1,
        inertia_local=compute_inertia(geometry, 1), is_static=static,
        restitution=restitution, friction=0,
    )
    result.linear_velocity = tuple(velocity)
    return result


def world(bodies, mode="spatial_hash", ground=None, budget=8):
    return PhysicsWorld(bodies, ground, SimulationSpec(
        enabled=True, physics_dt=.01, substeps=1, iterations=40,
        gravity=(0, 0, 0), linear_damping=0, angular_damping=0,
        baumgarte=0, slop=0, broadphase=mode, max_events_per_substep=budget,
    ))


def hull(size):
    return ConvexHullGeometry([
        (sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2)
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])


class CollisionReliabilityTests(unittest.TestCase):
    def test_support_transform_cache_tracks_pose_translation_and_geometry(self):
        moving = body("shape", CylinderGeometry(.2, .8), (1, 2, 3))
        direction = (.3, -.7, 1.2)
        for angle in (0, .4, 1.1):
            moving.orientation = quat.quat_from_axis_angle((1, 2, 3), angle)
            for position in ((1, 2, 3), (-2, 1, .5)):
                moving.position = position
                for geometry in (CylinderGeometry(.2, .8), SphereGeometry(.4)):
                    moving.geometry = geometry
                    local_direction = quat.quat_rotate_vector(quat.quat_conjugate(moving.orientation), direction)
                    point = quat.quat_rotate_vector(moving.orientation, geometry.support(local_direction))
                    expected = tuple(a + b for a, b in zip(position, point))
                    self.assertEqual(support_world(moving, direction), expected)
                    self.assertEqual(support_world(moving, direction), expected)

    def test_ground_culling_keeps_fast_falls_and_rotational_sweeps(self):
        floor = GroundPlane(z=0)
        far = body("far", BoxGeometry((.5, .5, .5)), (0, 0, 10))
        self.assertIsNone(ground_toi(far, floor, .01))
        far.linear_velocity = (0, 0, -1000)
        self.assertIsNotNone(ground_toi(far, floor, .02))
        rod = body("rod", BoxGeometry((2, .1, .1)), (0, 0, .6))
        rod.angular_velocity = (0, 20, 0)
        self.assertIsNotNone(ground_toi(rod, floor, .1))

    def test_conservative_probes_have_separate_work_budget(self):
        # The cube's XY circumradius is < .71, so it cannot hit this small sphere.
        # Its loose |omega|R bound still produces several conservative probes.
        for spin in (100, 10000):
            with self.subTest(spin=spin):
                spinner = body("spinner", BoxGeometry((1, 1, 1)), (0, 0, 0))
                spinner.angular_velocity = (0, 0, spin)
                target = body("target", SphereGeometry(.01), (.8, 0, 0), static=True)
                sim = world([spinner, target], budget=1)
                sim.simulation.physics_dt = .03
                if spin == 100:
                    sim.step()
                    self.assertAlmostEqual(sim.time, .03)
                else:
                    with self.assertRaisesRegex(RuntimeError, "advancement budget"):
                        sim.step()
                self.assertEqual(sim.last_event_count, 0)

    def test_sat_waits_for_all_separating_axes(self):
        moving = body("moving", BoxGeometry((1, 1, 1)), (1.02, 1.2, 0), (-1, -1, 0))
        wall = body("wall", BoxGeometry((1, 1, 1)), (0, 0, 0), static=True)
        self.assertAlmostEqual(body_toi(moving, wall, .3), .2002)
        sim = world([moving, wall], budget=1)
        sim.simulation.physics_dt = .3
        sim.step()
        self.assertEqual(sim.last_event_count, 1)
        self.assertAlmostEqual(sim.time, .3)

        # A positive gap on an axis with no closing motion rules out a collision.
        moving = body("moving", BoxGeometry((1, 1, 1)), (1.02, 1.2, 0), (0, -1, 0))
        self.assertIsNone(body_toi(moving, wall, .3))

    def test_rotated_thin_wall(self):
        for geometry in (BoxGeometry((.02, 2, 2)), hull((.02, 2, 2))):
            for angle in (math.pi / 6, 1.1):
                for gap in (0, 5e-8, 5e-7):
                    with self.subTest(geometry=geometry.type_name, angle=angle, gap=gap):
                        normal = (math.cos(angle), math.sin(angle), 0)
                        ball = body("ball", SphereGeometry(.1),
                                    tuple((.11 + gap) * n for n in normal),
                                    tuple(-100 * n for n in normal))
                        wall = body("wall", geometry, (0, 0, 0), static=True)
                        rotation = quat.quat_from_axis_angle((0, 0, 1), angle)
                        wall.orientation = quat.quat_multiply(rotation, wall.orientation)
                        world([ball, wall]).step()
                        self.assertGreater(sum(p * n for p, n in zip(ball.position, normal)), .109)
                        self.assertLess(sum(v * v for v in ball.linear_velocity), 1e-8)

    def test_sphere_thin_wall_near_contact(self):
        for mode in ("spatial_hash", "brute"):
            for wall_geometry in (BoxGeometry((.02, 2, 2)), hull((.02, 2, 2))):
                for gap in (0, 5e-12, 5e-8, 5e-7, 2e-6):
                    with self.subTest(mode=mode, wall=wall_geometry.type_name, gap=gap):
                        ball = body("ball", SphereGeometry(.1), (.11 + gap, 0, 0), (-100, 0, 0))
                        wall = body("wall", wall_geometry, (0, 0, 0), static=True)
                        sim = world([ball, wall], mode)
                        sim.step()
                        self.assertGreater(ball.position[0], .109)
                        self.assertAlmostEqual(ball.linear_velocity[0], 0, places=5)

    def test_curved_pair_near_contact(self):
        for gap in (0, 5e-8, 5e-7, 2e-6):
            for reverse in (False, True):
                with self.subTest(gap=gap, reverse=reverse):
                    cylinder = body("cylinder", CylinderGeometry(.1, .2), (.11 + gap, 0, 0), (-100, 0, 0))
                    wall = body("wall", BoxGeometry((.02, 2, 2)), (0, 0, 0), static=True)
                    sim = world([wall, cylinder] if reverse else [cylinder, wall])
                    sim.step()
                    self.assertGreater(cylinder.position[0], .109)
                    self.assertAlmostEqual(cylinder.linear_velocity[0], 0, places=4)

    def test_polytope_pair_near_contact(self):
        for geometry in (BoxGeometry((.2, .2, .2)), hull((.2, .2, .2))):
            for gap in (0, 5e-8, 5e-7):
                with self.subTest(geometry=geometry.type_name, gap=gap):
                    moving = body("moving", geometry, (.11 + gap, 0, 0), (-100, 0, 0))
                    wall = body("wall", BoxGeometry((.02, 2, 2)), (0, 0, 0), static=True)
                    world([moving, wall]).step()
                    self.assertGreater(moving.position[0], .109)
                    self.assertAlmostEqual(moving.linear_velocity[0], 0, places=4)

    def test_ground_near_contact(self):
        for geometry in (SphereGeometry(.1), BoxGeometry((.2, .2, .2)),
                         hull((.2, .2, .2)), CylinderGeometry(.1, .2)):
            for gap in (0, 5e-12, 5e-8, 5e-7):
                with self.subTest(geometry=geometry.type_name, gap=gap):
                    moving = body("moving", geometry, (0, 0, .1 + gap), (0, 0, -100))
                    sim = world([moving], ground=GroundPlane(z=0, restitution=0, friction=0))
                    sim.step()
                    self.assertGreater(moving.position[2], .099)
                    self.assertAlmostEqual(moving.linear_velocity[2], 0, places=4)

    def test_spheres_touching_and_subpicosecond_toi(self):
        for angle in (0, .26, 1.1):
            normal = (math.cos(angle), math.sin(angle), 0)
            for gap in (0, 5e-12, 5e-7):
                with self.subTest(gap=gap, angle=angle):
                    moving = body("moving", SphereGeometry(.1),
                                  tuple((.2 + gap) * n for n in normal),
                                  tuple(-100 * n for n in normal))
                    wall = body("wall", SphereGeometry(.1), (0, 0, 0), static=True)
                    world([moving, wall]).step()
                    self.assertGreater(sum(p * n for p, n in zip(moving.position, normal)), .199)
                    self.assertLess(sum(v * v for v in moving.linear_velocity), 1e-8)

    def test_existing_contact_refreshes_swept_candidates(self):
        for mode in ("spatial_hash", "brute"):
            with self.subTest(mode=mode):
                balls = [body(str(i), SphereGeometry(.1), (x, 0, 0),
                              (100 if i == 0 else 0, 0, 0), restitution=1)
                         for i, x in enumerate((0, .1999, .7, 1.2))]
                sim = world(balls, mode)
                sim.validate_initial_state()  # 0.1 mm overlap is within scene tolerance.
                sim.step()
                for b in balls[:3]:
                    self.assertAlmostEqual(b.linear_velocity[0], 0, places=5)
                self.assertAlmostEqual(balls[-1].linear_velocity[0], 100, places=5)
                self.assertTrue(all(a.position[0] < b.position[0] for a, b in zip(balls, balls[1:])))

    def test_broadphases_match_with_default_toi(self):
        outputs = []
        for mode in ("spatial_hash", "brute"):
            balls = [body("a", SphereGeometry(.1), (-.5, 0, 0), (100, 0, 0), restitution=1),
                     body("b", SphereGeometry(.1), (0, 0, 0), restitution=1)]
            sim = world(balls, mode)
            sim.step()
            outputs.append([(b.position, b.linear_velocity) for b in balls])
        self.assertEqual(outputs[0], outputs[1])

    def test_budget_exhaustion_does_not_free_fly(self):
        balls = [body(str(i), SphereGeometry(.1), (i * .3, 0, 0),
                      (100 if i == 0 else 0, 0, 0), restitution=1) for i in range(4)]
        sim = world(balls, budget=1)
        with self.assertRaisesRegex(RuntimeError, "max_events_per_substep"):
            sim.step()
        self.assertEqual(sim.last_event_count, 1)
        self.assertLess(balls[1].position[0], balls[2].position[0])

    def test_budget_last_event_can_finish_if_no_more_hits(self):
        moving = body("moving", SphereGeometry(.1), (.5, 0, 0), (-100, 0, 0))
        wall = body("wall", SphereGeometry(.1), (0, 0, 0), static=True)
        sim = world([moving, wall], budget=1)
        sim.step()
        self.assertAlmostEqual(sim.time, .01)
        self.assertEqual(sim.last_event_count, 1)
        self.assertAlmostEqual(moving.linear_velocity[0], 0)

    def test_touching_but_separating_has_no_event(self):
        for geometry in (SphereGeometry(.1), BoxGeometry((.2, .2, .2))):
            with self.subTest(geometry=geometry.type_name):
                moving = body("moving", geometry, (.2, 0, 0), (100, 0, 0))
                wall = body("wall", BoxGeometry((.2, 2, 2)), (0, 0, 0), static=True)
                sim = world([moving, wall])
                sim.step()
                self.assertEqual(sim.last_event_count, 0)
                self.assertAlmostEqual(moving.linear_velocity[0], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
