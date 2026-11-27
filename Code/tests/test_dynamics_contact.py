"""Physical regressions for momentum integration and contact response."""
import math
import os
import sys
import unittest
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geom.geometry import BoxGeometry, SphereGeometry, CylinderGeometry
from src.geom.inertia import compute_inertia
from src.geom.transform import TransformSpec
from src.mathutils import quaternion as quat, vec3
from src.physics.body import RigidBody
from src.physics.world import PhysicsWorld
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.contact import ContactPoint, ContactManifold
from src.physics.solver import SequentialImpulseSolver
from src.scene.scene_spec import ObjectSpec, MaterialSpec, SimulationSpec


def make_body(name="body", size=(1, 2, 3), position=(0, 0, 0)):
    geometry = BoxGeometry(size)
    return RigidBody(name, geometry, position=position,
                     inertia_local=compute_inertia(geometry, 1), friction=.5, restitution=0)


def make_world(bodies, dt=.001, **overrides):
    options = dict(enabled=True, physics_dt=dt, substeps=1, iterations=20,
                   gravity=(0, 0, 0), linear_damping=0, angular_damping=0,
                   use_toi_splitting=False)
    options.update(overrides)
    return PhysicsWorld(bodies, None, SimulationSpec(**options))


def angular_momentum(body):
    omega = quat.quat_rotate_vector(quat.quat_conjugate(body.orientation), body.angular_velocity)
    return quat.quat_rotate_vector(body.orientation, tuple(i * w for i, w in zip(body.inertia_local, omega)))


class DynamicsContactTests(unittest.TestCase):
    def test_inverse_inertia_cache_tracks_pose_and_diagonal(self):
        body = make_body()
        rng = random.Random(43)
        for _ in range(30):
            body.orientation = quat.quat_from_axis_angle((1, 2, 3), rng.uniform(-3, 3))
            vector = tuple(rng.uniform(-4, 4) for _ in range(3))
            local = quat.quat_rotate_vector(quat.quat_conjugate(body.orientation), vector)
            expected = quat.quat_rotate_vector(body.orientation,
                       tuple(i * x for i, x in zip(body.inv_inertia_local, local)))
            actual = body.inverse_inertia_world(vector)
            self.assertLess(max(abs(a - b) for a, b in zip(actual, expected)), 1e-13)
            body.inverse_inertia_world(vector, orientation=quat.quat_identity())
            self.assertEqual(body.inverse_inertia_world(vector), actual)
        body.orientation = quat.quat_identity()
        body.inverse_inertia_world((1, 1, 1))
        body.inv_inertia_local = (2, 3, 4)
        self.assertEqual(body.inverse_inertia_world((1, 1, 1)), (2, 3, 4))

    def test_position_correction_cannot_increase_spinning_body_energy(self):
        rng = random.Random(17)
        for _ in range(40):
            body = make_body(position=(0, 0, 1.4))
            body.orientation = quat.quat_from_axis_angle((1, 2, 3), rng.random() * .5)
            body.angular_velocity = tuple(rng.uniform(-3, 3) for _ in range(3))
            sim = make_world([body])
            sim.ground = GroundPlane(z=0)
            energy, momentum = body.kinetic_energy(), body.angular_momentum
            sim.solver.solve_positions(detect_contacts(sim.bodies, sim.ground), sim.get_body, .001)
            self.assertLessEqual(body.kinetic_energy(), energy + 2e-12 * max(1., energy))
            self.assertEqual(body.angular_momentum, momentum)

    def test_curved_contact_preserves_both_surface_anchors(self):
        geometry = CylinderGeometry(1, 1)
        cylinder = RigidBody("cylinder", geometry, inertia_local=compute_inertia(geometry, 1))
        box = make_body("box", size=(1, 1, 1), position=(1.3, 0, .9))
        manifold = detect_contacts([cylinder, box], None)[0]
        for point in manifold.points:
            a = vec3.add(cylinder.position, quat.quat_rotate_vector(cylinder.orientation, point.local_anchor_a))
            b = vec3.add(box.position, quat.quat_rotate_vector(box.orientation, point.local_anchor_b))
            self.assertAlmostEqual(a[2], .5)
            self.assertAlmostEqual(b[2], .4)
            self.assertGreaterEqual(a[0], .8 - 1e-9)
            self.assertLessEqual(a[0], 1 + 1e-9)

    def test_warm_start_is_applied_once_per_substep(self):
        body = make_body(position=(0, 0, .5))
        sim = make_world([body])
        contact = ContactManifold(-1, 0, [ContactPoint((0, 0, 0), (0, 0, 1), 0)], friction=0, restitution=0)
        solver = sim.solver
        body.linear_velocity = (0, 0, -1)
        solver.begin_substep()
        solver.solve([contact], sim.get_body, .001)
        solver.begin_substep()
        body.linear_velocity = (0, 0, -1)
        solver.solve([contact], sim.get_body, .001)
        self.assertAlmostEqual(body.linear_velocity[2], 0)
        solver.iterations = 0
        solver.solve([contact], sim.get_body, .001)
        self.assertAlmostEqual(body.linear_velocity[2], 0)

    def test_penetration_correction_does_not_create_kinetic_energy(self):
        body = make_body(size=(1, 1, 1), position=(0, 0, .45))
        sim = make_world([body])
        sim.ground = GroundPlane(z=0, restitution=0, friction=0)
        before = sim.max_penetration()
        sim.step()
        self.assertLess(sim.max_penetration(), before)
        self.assertLess(body.kinetic_energy(), 1e-20)
        self.assertEqual(body.linear_velocity, (0, 0, 0))

    def test_restitution_is_independent_of_position_bias(self):
        speeds = []
        for beta in (0, .25):
            body = RigidBody("ball", SphereGeometry(.5), position=(0, 0, .49),
                             restitution=.5, friction=0)
            body.linear_velocity = (0, 0, -2)
            sim = make_world([body], baumgarte=beta)
            sim.ground = GroundPlane(z=0, restitution=.5, friction=0)
            sim.step()
            speeds.append(body.linear_velocity[2])
        self.assertAlmostEqual(speeds[0], 1)
        self.assertAlmostEqual(speeds[1], 1)

    def test_position_solve_preserves_physical_momenta(self):
        body = make_body(size=(1, 1, 1), position=(0, 0, .49))
        body.angular_velocity = (1, 2, 3)
        body.linear_velocity = (1, 0, 0)
        sim = make_world([body])
        sim.ground = GroundPlane(z=0)
        momentum, velocity = body.angular_momentum, body.linear_velocity
        sim.solver.solve_positions(detect_contacts(sim.bodies, sim.ground), sim.get_body, .001)
        self.assertEqual(body.angular_momentum, momentum)
        self.assertEqual(body.linear_velocity, velocity)

    def test_contact_cache_expires_during_free_flight(self):
        body = make_body(position=(0, 0, .5))
        sim = make_world([body])
        solver = sim.solver
        body.linear_velocity = (0, 0, -1)
        contact = ContactManifold(-1, 0, [ContactPoint((0, 0, 0), (0, 0, 1), 0)], friction=0, restitution=0)
        solver.begin_substep()
        solver.solve([contact], sim.get_body, .001)
        self.assertTrue(solver._impulse_cache)
        for _ in range(4):
            solver.begin_substep()
        self.assertFalse(solver._impulse_cache)

    def test_cache_checks_anchors_normals_and_feature_switches(self):
        for mode in ("near_new_id", "far_same_id", "changed_normal"):
            with self.subTest(mode=mode):
                body = make_body(position=(0, 0, .5))
                sim = make_world([body])
                solver = sim.solver
                body.linear_velocity = (0, 0, -1)
                initial = ContactManifold(-1, 0, [ContactPoint((0, 0, 0), (0, 0, 1), 0)], friction=0, restitution=0)
                solver.begin_substep()
                solver.solve([initial], sim.get_body, .001)
                body.linear_velocity = (0, 0, 0)
                solver.begin_substep()
                position = (.001, 0, 0) if mode == "near_new_id" else (1, 0, 0) if mode == "far_same_id" else (0, 0, 0)
                normal = (0, 1, 0) if mode == "changed_normal" else (0, 0, 1)
                point = ContactPoint(position, normal, 0, feature_id=99 if mode == "near_new_id" else 0)
                solver.iterations = 0  # Observe only the cache's physical response.
                solver.solve([ContactManifold(-1, 0, [point], friction=0, restitution=0)], sim.get_body, .001)
                self.assertAlmostEqual(body.linear_velocity[2], 1 if mode == "near_new_id" else 0)
                self.assertIsNotNone(point.local_anchor_a)
                self.assertIsNotNone(point.local_anchor_b)

    def test_ground_uses_both_materials(self):
        body = make_body(position=(0, 0, .49), size=(1, 1, 1))
        body.friction = 0
        body.restitution = .9
        sim = make_world([body])
        sim.ground = GroundPlane(z=0, friction=.5, restitution=.2)
        contact = detect_contacts(sim.bodies, sim.ground)[0]
        self.assertEqual(contact.friction, 0)
        self.assertEqual(contact.restitution, .9)

    def test_friction_is_isotropic_and_bounded(self):
        for angle in (0, .3, math.pi / 4, 1.2):
            with self.subTest(angle=angle):
                body = make_body(size=(1, 1, 1), position=(0, 0, .4999))
                before = (10 * math.cos(angle), 10 * math.sin(angle), 0)
                body.linear_velocity = before
                sim = make_world([body], iterations=80, gravity=(0, 0, -9.81))
                sim.ground = GroundPlane(z=0, friction=.5, restitution=0)
                sim.step()
                normal = body.linear_velocity[2] + .00981
                tangent = math.hypot(body.linear_velocity[0] - before[0], body.linear_velocity[1] - before[1])
                self.assertAlmostEqual(tangent / normal, .5, places=8)
                self.assertLess(body.kinetic_energy(), 50)

    def test_explicit_inertia_preserves_input_orientation(self):
        geometry = BoxGeometry((1, 2, 3))
        q = quat.quat_from_axis_angle((1, 2, 3), .7)
        spec = ObjectSpec("asset", geometry, TransformSpec(orientation=q), MaterialSpec(),
                          mass=1, inertia=compute_inertia(geometry, 1))
        body = RigidBody.from_object_spec(spec)
        self.assertAlmostEqual(abs(quat.quat_dot(q, body.orientation)), 1, places=12)

    def test_torque_free_rotation_conserves_world_momentum(self):
        body = make_body()
        body.angular_velocity = (1, 2, 3)
        initial = angular_momentum(body)
        sim = make_world([body])
        for _ in range(1000):
            sim.step()
        drift = vec3.length(vec3.sub(angular_momentum(body), initial)) / vec3.length(initial)
        self.assertLess(drift, 1e-10)
        self.assertGreater(vec3.length(vec3.sub(body.angular_velocity, (1, 2, 3))), .1)

    def test_rotation_energy_error_converges(self):
        errors = []
        for dt in (.02, .01):
            body = make_body()
            body.angular_velocity = (1, 2, 3)
            energy = body.kinetic_energy()
            sim = make_world([body], dt)
            for _ in range(round(1 / dt)):
                sim.step()
            errors.append(abs(body.kinetic_energy() - energy) / energy)
        self.assertLess(errors[0], .005)
        self.assertLess(errors[1], errors[0] * .4)

    def test_force_torque_and_impulse_have_consistent_units(self):
        body = make_body()
        sim = make_world([body], .01, substeps=4)
        body.add_force((2, 0, 0))
        body.add_torque((0, 0, 3))
        sim.step()
        self.assertAlmostEqual(body.linear_velocity[0], .02)
        self.assertAlmostEqual(body.angular_momentum[2], .03)
        sim.step()  # force accumulators must have been cleared once per full step.
        self.assertAlmostEqual(body.linear_velocity[0], .02)
        self.assertAlmostEqual(body.angular_momentum[2], .03)
        old = body.angular_momentum
        body.apply_impulse((0, 2, 0), (1, 0, 0))
        self.assertAlmostEqual(body.angular_momentum[2] - old[2], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
