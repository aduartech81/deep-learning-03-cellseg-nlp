"""Pruebas de las utilidades metodológicas (incluye las verificaciones sintéticas de 4.8.24 y casos borde)."""
import numpy as np
import pandas as pd
import pytest

from dl03 import calibration as cal
from dl03 import drift, stats
from dl03 import seg_metrics as sm
from dl03.rle import instances_to_label_map, label_map_to_rles, rle_decode, rle_encode

rng = np.random.default_rng(0)


def test_rle_roundtrip_and_edges():
    m = np.zeros((40, 50), np.uint8)
    m[5:15, 10:30] = 1
    m[25:35, 5:20] = 1
    assert (rle_decode(rle_encode(m), m.shape) == m).all()
    assert rle_decode("", (3, 3)).sum() == 0 and rle_decode(float("nan"), (3, 3)).sum() == 0
    with pytest.raises(ValueError):
        rle_decode("8 5", (3, 3))
    # 1-indexado y orden de filas: "1 3" = primera fila completa de una imagen 3x3
    assert (rle_decode("1 3", (3, 3))[0] == 1).all()


def test_label_map_overlap_first_wins():
    a = rle_encode(np.pad(np.ones((3, 3)), ((0, 3), (0, 3))))
    b = rle_encode(np.pad(np.ones((3, 3)), ((2, 1), (2, 1))))
    lbl, lost = instances_to_label_map([a, b], (6, 6), return_overlap=True)
    assert lost == 1 and lbl[2, 2] == 1 and (lbl == 2).sum() == 8
    assert len(label_map_to_rles(lbl)) == 2


def _two_squares():
    gt = np.zeros((64, 64), np.int32)
    gt[5:25, 5:25] = 1
    gt[35:60, 35:60] = 2
    return gt


def test_instance_metrics_known_cases():
    gt = _two_squares()
    assert abs(sm.sartorius_score(gt, gt) - 1.0) < 1e-12
    assert sm.sartorius_score(gt, np.zeros_like(gt)) == 0.0
    pq, sq, rq = sm.panoptic_quality(gt, gt)
    assert abs(pq - 1) < 1e-12
    assert abs(sm.aji(gt, gt) - 1) < 1e-12
    assert sm.boundary_f1(gt, gt) == 1.0
    # una sola predicción que cubre ambas = fusión de 2 referencias
    merged = (gt > 0).astype(np.int32)
    merged[20:40, 20:40] = 1
    tax = sm.error_taxonomy(gt, merged)
    assert tax["fusionada"] == 2 and tax["falsa"] == 0
    # dividir el cuadrado grande en tres franjas (ninguna con IoU > 0.5) = 1 referencia dividida
    split = gt.copy()
    split[35:60, 43:51] = 3
    split[35:60, 51:60] = 4
    assert sm.error_taxonomy(gt, split)["dividida"] == 1
    # predicción vacía: todo omitido
    assert sm.error_taxonomy(gt, np.zeros_like(gt))["omitida"] == 2
    row = sm.image_metrics(gt, gt, prob=(gt > 0).astype(float))
    assert row["ap"] == 1.0 and row["auc_roc"] == 1.0 and row["inst_f1"] == 1.0


def test_aji_penalises_extra_prediction():
    gt = _two_squares()
    pr = gt.copy()
    pr[0:3, 0:3] = 9
    assert sm.aji(gt, pr) < 1.0


def test_dice_iou_relation():
    a = rng.random((64, 64)) > 0.5
    b = rng.random((64, 64)) > 0.5
    d, j = sm.dice_iou(a, b)
    assert abs(d - 2 * j / (1 + j)) < 1e-6


def test_hausdorff95_shift():
    s1 = np.zeros((60, 60), bool)
    s1[10:40, 10:40] = True
    s2 = np.zeros((60, 60), bool)
    s2[10:40, 13:43] = True
    assert 2.0 <= sm.hausdorff95(s1, s2) <= 4.0


def test_compare_models_and_holm():
    n = 80
    base = rng.normal(0.6, 0.05, n)
    sc = pd.DataFrame({"M1": base, "M2": base + 0.05 + rng.normal(0, 0.01, n), "M3": base + rng.normal(0, 0.01, n)})
    res = stats.compare_models(sc)
    assert res["friedman_p"] < 0.05
    pw = res["pairwise"].set_index(["A", "B"])
    assert pw.loc[("M1", "M2"), "reject"] and not pw.loc[("M1", "M3"), "reject"]
    mean, lo, hi = stats.bootstrap_ci(sc["M2"], n_boot=2000)
    assert lo < mean < hi
    assert np.allclose(stats.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_nemenyi_cd_positive():
    assert 0 < stats.nemenyi_cd(5, 90) < 1


def test_temperature_scaling():
    z = rng.normal(0, 1, (4000, 3))
    p_true = cal.softmax(z)
    y = np.array([rng.choice(3, p=p) for p in p_true])
    logits = 3 * z
    T = cal.fit_temperature(logits, y)
    assert 2.0 < T < 4.0
    assert cal.expected_calibration_error(cal.softmax(logits / T), y) < cal.expected_calibration_error(
        cal.softmax(logits), y)
    assert (cal.softmax(logits / T).argmax(1) == cal.softmax(logits).argmax(1)).all()


def test_conformal_coverage():
    z = rng.normal(0, 1, (6000, 4))
    p = cal.softmax(2 * z)
    y = np.array([rng.choice(4, p=q) for q in p])
    sets, _ = cal.conformal_lac(p[:3000], y[:3000], p[3000:], alpha=0.1)
    cov = sets[np.arange(3000), y[3000:]].mean()
    assert 0.87 <= cov <= 0.93


def test_drift_psi_and_mmd():
    ref = rng.normal(0, 1, 5000)
    assert drift.psi(ref, rng.normal(0, 1, 5000)) < 0.05 and drift.psi(ref, rng.normal(1, 1, 5000)) > 0.25
    _, p_same = drift.mmd_permutation_test(rng.normal(0, 1, (150, 2)), rng.normal(0, 1, (150, 2)), n_perm=100)
    _, p_diff = drift.mmd_permutation_test(rng.normal(0, 1, (150, 2)), rng.normal(1, 1, (150, 2)), n_perm=100)
    assert p_same > 0.05 and p_diff < 0.05
