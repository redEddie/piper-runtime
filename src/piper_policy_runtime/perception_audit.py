"""Camera-only measurement reporting; no robot SDK, policy or device imports."""
import numpy as np

from .show100 import vector


def validate_calibration(document):
    transform = np.asarray(document['T_base_cam'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
        raise ValueError('T_base_cam must be a finite rigid transform')
    if document.get('frame') != 'piper_base_link':
        raise ValueError('Calibration must use piper_base_link')
    return transform


class PerceptionAudit:
    def __init__(self, object_ids, *, max_age_s=.2, stationary=False, reference=None,
                 reference_tolerance_m=.01):
        self.ids = tuple(object_ids)
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Object IDs must be nonempty and unique')
        if not np.isfinite([max_age_s, reference_tolerance_m]).all() or min(max_age_s, reference_tolerance_m) <= 0:
            raise ValueError('Age and tolerance must be positive and finite')
        self.max_age_s, self.stationary = max_age_s, stationary
        self.tolerance = reference_tolerance_m
        self.reference = {}
        if reference is not None:
            if reference.get('frame') != 'piper_base_link' or reference.get('units') != 'm':
                raise ValueError('Reference must use piper_base_link and metres')
            for obj in reference['objects']:
                if obj['id'] in self.reference:
                    raise ValueError('Duplicate reference object')
                self.reference[obj['id']] = vector(obj['pos'], 3, 'reference position')
        self.polls = 0
        self.fresh = {oid: 0 for oid in self.ids}
        self.samples = {oid: {} for oid in self.ids}
        self.ages = {oid: [] for oid in self.ids}

    def add(self, now, objects, *, offline=False):
        """Each object has pos, frame, t. Repeated stale estimates never add samples."""
        self.polls += 1
        fresh = []
        for oid in self.ids:
            obj = objects.get(oid)
            if obj is None:
                continue
            pos = vector(obj['pos'], 3, f'{oid} position')
            if not offline:
                age = float(now - obj['t'])
                if not np.isfinite(age) or age < 0:
                    continue
                self.ages[oid].append(age)
                if age > self.max_age_s:
                    continue
            self.fresh[oid] += 1
            self.samples[oid][obj['frame']] = pos
            fresh.append(oid)
        return fresh

    def summary(self):
        objects = {}
        for oid in self.ids:
            points = np.array(list(self.samples[oid].values()))
            entry = dict(unique_measurements=len(points),
                         fresh_poll_fraction=self.fresh[oid] / max(self.polls, 1),
                         age_p95_ms=float(np.percentile(self.ages[oid], 95)*1000) if self.ages[oid] else None)
            if len(points):
                median = np.median(points, axis=0)
                entry['median_position_base_m'] = median.tolist()
                if self.stationary and len(points) >= 2:
                    entry['stationary_radial_jitter_p95_mm'] = float(np.percentile(np.linalg.norm(points-median, axis=1), 95)*1000)
                if oid in self.reference:
                    error = median-self.reference[oid]
                    entry.update(reference_error_xyz_mm=(error*1000).tolist(),
                                 reference_error_3d_mm=float(np.linalg.norm(error)*1000),
                                 within_reference_tolerance=bool(np.linalg.norm(error) <= self.tolerance))
            objects[oid] = entry
        return dict(polls=self.polls, objects=objects,
                    stationary_scene_declared=self.stationary,
                    max_age_s=self.max_age_s, reference_tolerance_m=self.tolerance,
                    independent_reference_supplied=bool(self.reference),
                    absolute_accuracy_status='compared_to_supplied_reference' if self.reference else 'unverified_no_independent_reference',
                    hardware_commands_sent=False)
