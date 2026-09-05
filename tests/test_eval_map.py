"""scripts/eval_map.py's metric math - fabricated boxes/scores only, no real
predictions or ground truth ever touch this file."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.eval_map import average_precision, evaluate, iou_matrix


def test_iou_matrix_identical_boxes_is_one():
    det = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(det, gt)[0, 0] == pytest.approx(1.0)


def test_iou_matrix_disjoint_boxes_is_zero():
    det = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt = np.array([[100, 100, 110, 110]], dtype=np.float32)
    assert iou_matrix(det, gt)[0, 0] == 0.0


def test_iou_matrix_empty_inputs_shape():
    empty = np.zeros((0, 4), dtype=np.float32)
    one = np.array([[0, 0, 1, 1]], dtype=np.float32)
    assert iou_matrix(empty, one).shape == (0, 1)
    assert iou_matrix(one, empty).shape == (1, 0)


def test_average_precision_zero_ground_truth_is_nan_not_zero():
    """A class with no instances must produce NaN, not 0.0 - conflating the two
    is exactly the bug that would make an unvalidatable class silently drag mAP
    toward zero instead of being excluded from it (see app's `other` class)."""
    ap, p, r = average_precision(np.array([], dtype=np.float32), np.array([], dtype=np.float32), n_gt=0)
    assert np.isnan(ap) and np.isnan(p) and np.isnan(r)


def test_average_precision_no_detections_but_instances_exist():
    ap, p, r = average_precision(np.array([], dtype=np.float32), np.array([], dtype=np.float32), n_gt=5)
    assert (ap, p, r) == (0.0, 0.0, 0.0)


def test_average_precision_perfect_detector_is_near_one():
    tp = np.array([1, 1, 1], dtype=np.float32)
    conf = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    ap, precision, recall = average_precision(tp, conf, n_gt=3)
    assert ap > 0.99
    assert precision == 1.0
    assert recall == 1.0


def test_average_precision_all_false_positives_is_zero():
    tp = np.array([0, 0, 0], dtype=np.float32)
    conf = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    ap, precision, _ = average_precision(tp, conf, n_gt=3)
    assert ap == 0.0
    assert precision == 0.0


def _empty_record():
    return {
        "det_boxes": np.zeros((0, 4), np.float32),
        "det_scores": np.zeros((0,), np.float32),
        "det_classes": np.zeros((0,), np.int64),
        "gt_boxes": np.zeros((0, 4), np.float32),
        "gt_classes": np.zeros((0,), np.int64),
    }


def test_evaluate_zero_instance_class_does_not_corrupt_overall_map():
    """The `other`-class problem in miniature: class 4 has zero ground truth
    anywhere in the (fabricated) dataset. Its AP must be NaN, and it must be
    excluded from the aggregate rather than silently counted as 0."""
    per_class, overall = evaluate([_empty_record()], class_ids=[0, 4])
    assert np.isnan(per_class[4]["ap50_95"])
    assert per_class[4]["instances"] == 0
    # no class has any instances in this fabricated record, so the aggregate
    # is defined as 0.0 over an empty "present" set, not NaN and not corrupted
    assert overall["map50_95"] == 0.0
