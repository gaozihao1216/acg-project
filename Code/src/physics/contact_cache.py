"""Persistent contact matching in body coordinates, independent of solver iterations."""
from dataclasses import dataclass

from ..mathutils import quaternion as quat, vec3


def local_point(body, point):
    return quat.quat_rotate_vector(quat.quat_conjugate(body.orientation), vec3.sub(point, body.position))


def world_point(body, point):
    return vec3.add(body.position, quat.quat_rotate_vector(body.orientation, point))


def ensure_anchors(point, a, b):
    """Populate geometry anchors without consulting or changing warm-start history."""
    if point.local_anchor_a is None:
        point.local_anchor_a = local_point(a, vec3.madd(point.position, point.normal, .5 * point.penetration))
    if point.local_anchor_b is None:
        point.local_anchor_b = local_point(b, vec3.madd(point.position, point.normal, -.5 * point.penetration))


@dataclass
class CachedContact:
    key: tuple
    body_a: object
    body_b: object
    anchor_a: tuple
    anchor_b: tuple
    normal: tuple
    normal_impulse: float
    h: float
    seen_epoch: int
    warm_epoch: int = -1


class ContactManager:
    """Only adjacent substeps may reuse impulses; all matches are one-to-one.

    A feature ID is a search hint, not proof of physical contact identity.
    Surface anchors are reconstructed from the midpoint and normal separation
    when a narrowphase backend does not supply them explicitly.
    """

    def __init__(self, slop):
        self.slop = slop
        self.epoch = 0
        self.entries = {}
        self.claimed = set()
        self.by_pair = {}

    def clear(self):
        self.entries.clear()
        self.claimed.clear()
        self.by_pair.clear()

    def begin_substep(self):
        self.epoch += 1
        # Advance even during free flight, when the velocity solver is never called.
        stale = [key for key, entry in self.entries.items() if entry.seen_epoch < self.epoch - 1]
        for key in stale:
            del self.entries[key]

    def begin_query(self):
        self.claimed.clear()
        self.by_pair = {}
        for key, entry in self.entries.items():
            self.by_pair.setdefault(key[:2], []).append((key, entry))

    def bind(self, point, key, a, b):
        ensure_anchors(point, a, b)
        radii = [body.bounding_radius for body in (a, b) if body.geometry is not None]
        tolerance = max(1e-5, min(radii, default=1) * .04)
        best = None
        best_score = float("inf")
        for old_key, entry in self.by_pair.get(key[:2], ()):
            if id(entry) in self.claimed:
                continue
            if entry.body_a is not a or entry.body_b is not b:
                continue
            if entry.seen_epoch < self.epoch - 1 or vec3.dot(entry.normal, point.normal) < .95:
                continue
            da = vec3.length_squared(vec3.sub(entry.anchor_a, point.local_anchor_a))
            db = vec3.length_squared(vec3.sub(entry.anchor_b, point.local_anchor_b))
            if max(da, db) > tolerance * tolerance:
                continue
            gap = vec3.dot(vec3.sub(world_point(b, entry.anchor_b), world_point(a, entry.anchor_a)), point.normal)
            if gap > max(self.slop * 2, tolerance * .1):
                continue
            score = da + db
            if score < best_score or (score == best_score and old_key == key):
                best, best_score = entry, score
        if best is not None:
            self.claimed.add(id(best))
        return best

    def store(self, constraints, h):
        updates = {}
        for c in constraints:
            old = c.cache_entry
            impulse = c.accumulated_normal
            if old is not None and old.seen_epoch == self.epoch:
                impulse += old.normal_impulse
            updates[c.key] = CachedContact(
                c.key, c.body_a, c.body_b, c.anchor_a, c.anchor_b, c.normal,
                impulse, h, self.epoch, self.epoch,
            )
        for c in constraints:
            old = c.cache_entry
            if old is not None and self.entries.get(old.key) is old:
                del self.entries[old.key]
        self.entries.update(updates)
