"""Pruebas de datos (pandera), particiones sin fuga, geometría y objetivos de entrenamiento."""
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from dl03 import nlp_data as nd
from dl03 import seg_data as sd
from dl03.config import load_config, set_global_seed


@pytest.fixture(scope="module")
def cfg():
    return load_config("smoke")


@pytest.fixture(scope="module")
def seg(cfg):
    return sd.build_index(cfg)


def test_seg_schema_and_integrity(cfg, seg):
    images, instances = seg
    ann = pd.read_csv(cfg.data_dir / "raw" / "sartorius" / "train.csv")
    assert list(ann.columns) == sd.SEG_COLUMNS
    pa.DataFrameSchema({"id": pa.Column(str, unique=True), "n_inst": pa.Column(int, pa.Check.ge(1)),
                        "coverage": pa.Column(float, pa.Check.in_range(0, 1))}).validate(images)
    assert (instances["area"] > 0).all()


def test_seg_splits_no_leak(cfg, seg):
    images, _ = seg
    sp = sd.make_splits(images, cfg)
    sd.assert_no_group_leak(images, sp)
    assert set(sp.train) | set(sp.val) | set(sp.test) == set(images.id)
    for tr, te in sp.folds:
        assert not set(tr) & set(te) and not set(te) & set(sp.test)


def test_geometry_roundtrip():
    lbl = np.zeros((520, 704), np.int32)
    lbl[100:140, 200:260] = 3
    for res in [256, 512, "native"]:
        g = sd.Geometry.for_resolution(lbl.shape, res)
        m = g.to_model(lbl, order=0)
        assert m.shape == g.padded and m.shape[0] % 32 == 0 and m.shape[1] % 32 == 0
        back = g.to_native(m, order=0)
        assert back.shape == lbl.shape
        assert (back == 3).sum() > 0.8 * (lbl == 3).sum()


def test_targets():
    lbl = np.zeros((40, 40), np.int32)
    lbl[5:20, 5:20] = 1
    lbl[5:20, 20:35] = 2
    t = sd.three_class_target(lbl)
    assert set(np.unique(t)) == {0, 1, 2} and (t[lbl == 0] == 0).all()
    hv = sd.hv_maps(lbl)
    assert hv.min() >= -1 - 1e-6 and hv.max() <= 1 + 1e-6 and (hv[lbl == 0] == 0).all()


def test_nlp_cleaning_dedup_and_splits(cfg):
    df = nd.load_resumes(cfg)
    pa.DataFrameSchema({"text": pa.Column(str, pa.Check.str_length(min_value=1)),
                        "label": pa.Column(str, pa.Check.isin(nd.CATEGORIES))}).validate(df)
    assert nd.clean_standard("The Manager, managed 3 teams!") == "manager managed 3 teams"
    assert "manag" in nd.clean_aggressive("Managing managers at www.x.com 2019")
    groups, pairs = nd.find_duplicates(df["text"].tolist())
    assert any(p[3] for p in pairs), "el sintético incluye un duplicado exacto que debe detectarse"
    sp = nd.make_splits(df, groups, cfg)
    nd.assert_no_leak(groups, sp)
    assert nd.mtld("a b c d a b c d e f".split()) > 0


def test_seed_determinism():
    set_global_seed(7)
    a = np.random.rand(5)
    set_global_seed(7)
    assert np.allclose(a, np.random.rand(5))
