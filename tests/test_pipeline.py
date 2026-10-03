"""End-to-end tests on synthetic DICOM-RT data (CPU, ~1 min)."""
import os

import numpy as np
import pytest
import torch

from dosevlm import dicom_io
from dosevlm.dataset import DoseDataset, collate, num_input_channels
from dosevlm.engine import load_checkpoint, predict_sample, train
from dosevlm.export import write_rtdose
from dosevlm.geometry import Geometry, make_target_grid, resample
from dosevlm.losses import DoseLoss, dvh_loss
from dosevlm.metrics import gamma_pass_rate
from dosevlm.model import DoseVLM
from dosevlm.preprocess import load_sample, preprocess_patient
from dosevlm.rtplan import load_rtplan
from dosevlm.structures import classify, target_dose_from_name
from dosevlm.synthetic import make_patient
from dosevlm.text import SimpleTokenizer

PP = dict(spacing=(4.0, 4.0, 4.0), shape=(32, 48, 48), max_cp_tokens=32)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    npz = root / "npz"
    for i in range(4):
        d = root / f"P{i}"
        make_patient(str(d), seed=i, n_xy=64, n_z=32)
        preprocess_patient(str(d), str(npz / f"P{i}.npz"), **PP)
    return root


def test_geometry_roundtrip():
    g = Geometry([-10, 5, 3], (1.5, 2.0, 3.0), (4, 5, 6))
    ijk = np.random.rand(10, 3) * 4
    assert np.allclose(g.world_to_index(g.index_to_world(ijk)), ijk)


def test_resample_identity_and_shift():
    g = make_target_grid([0, 0, 0], (2, 2, 2), (10, 12, 14))
    v = np.random.rand(*g.shape).astype(np.float32)
    assert np.allclose(resample(v, g, g), v, atol=1e-5)
    g2 = Geometry(g.origin + [2, 0, 0], g.spacing, g.shape)  # shift one voxel in x
    assert np.allclose(resample(v, g, g2)[:, :, :-1], v[:, :, 1:], atol=1e-5)


def test_target_dose_parsing():
    assert target_dose_from_name("PTV_70", None) == 70
    assert target_dose_from_name("PTV7000", None) == 70
    assert target_dose_from_name("PTV_54.0Gy", None) == 54
    assert target_dose_from_name("PTV_High", 66.0) == 66


def test_dicom_readers(cohort):
    files = dicom_io.scan_patient_dir(str(cohort / "P0"))
    ct, geom, _ = dicom_io.load_ct(files.ct)
    assert ct.shape == (32, 64, 64) and ct.min() >= -1024
    masks = dicom_io.load_rtstruct(files.rtstruct[0], geom)
    body, targets, oars = classify(masks, None)
    assert body is not None and targets and "SpinalCord" in oars
    plan = load_rtplan(files.rtplan[0])
    assert plan.prescription_gy and plan.beams and all(b.is_arc for b in plan.beams)
    assert abs(sum(cp.mu for b in plan.beams for cp in b.control_points) - plan.total_mu) < 1e-3
    dose, dgeom, _ = dicom_io.load_rtdose(files.rtdose[0])
    on_ct = resample(dose, dgeom, geom)
    ptv = targets[0][2]
    assert on_ct[ptv].mean() > 0.5 * plan.prescription_gy


def test_rtdose_export_roundtrip(tmp_path):
    g = make_target_grid([1, 2, 3], (2.5, 2.5, 3.0), (6, 8, 10))
    d = np.random.rand(*g.shape).astype(np.float32) * 70
    p = write_rtdose(d, g, str(tmp_path / "rd.dcm"), meta={"plan_uid": "1.2.3"})
    d2, g2, ds = dicom_io.load_rtdose(p)
    assert np.allclose(d2, d, atol=1e-4) and np.allclose(g2.origin, g.origin) and np.allclose(g2.spacing, g.spacing)
    assert ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == "1.2.3"


def test_model_forward_backward_odd_shape():
    tok = SimpleTokenizer.build(["VMAT plan 70 Gy 2 arcs"])
    m = DoseVLM(num_input_channels(3), tok.vocab_size, dim=32, channels=(4, 8, 16), fusion_depth=1, heads=2,
                plan_depth=1, text_depth=1)
    ids, mask = tok.batch(["VMAT plan 70 Gy", "2 arcs"])
    x = torch.randn(2, num_input_channels(3), 13, 18, 21)
    out = m(x, torch.randn(2, 10), torch.rand(2, 16, 12), ids, mask)
    assert out.shape == (2, 1, 13, 18, 21) and (out >= 0).all()
    tgt = torch.rand_like(out)
    struct = torch.rand(2, 2, 13, 18, 21) > 0.7
    loss, terms = DoseLoss()(out, tgt, torch.ones_like(out, dtype=torch.bool), struct)
    loss.backward()
    assert set(terms) == {"l1", "ptv", "dvh", "grad"}
    assert all(p.grad is not None for p in m.head.parameters())


def test_dvh_loss_zero_for_permutation():
    t = torch.rand(1, 1, 4, 4, 4)
    s = torch.ones(1, 1, 4, 4, 4, dtype=torch.bool)
    perm = t.flatten()[torch.randperm(64)].view_as(t)
    assert float(dvh_loss(perm, t, s, n_samples=10_000)) < 1e-6


def test_gamma():
    ref = np.random.rand(8, 8, 8).astype(np.float32) + 1
    assert gamma_pass_rate(ref, ref, (3, 3, 3)) == 1.0
    assert gamma_pass_rate(ref * 1.5, ref, (3, 3, 3)) < 0.5


def test_dataset_and_train_predict(cohort, tmp_path):
    files = sorted(str(p) for p in (cohort / "npz").glob("*.npz"))
    ds = DoseDataset(files, patch_size=(16, 32, 32), train=True)
    b = collate([ds[0], ds[1]])
    assert b["image"].shape == (2, num_input_channels(len(load_sample(files[0])["oar_names"])), 16, 32, 32)

    cfg = {
        "preprocess": {**{k: list(v) if isinstance(v, tuple) else v for k, v in PP.items()}},
        "data": {"train": [str(cohort / "npz")], "val_fraction": 0.25, "patch_size": None},
        "model": {"dim": 32, "channels": [4, 8, 16], "fusion_depth": 1, "heads": 2, "plan_depth": 1,
                  "text_depth": 1},
        "train": {"output_dir": str(tmp_path / "run"), "epochs": 3, "batch_size": 2, "lr": 1e-3,
                  "num_workers": 0, "amp": False, "warmup_steps": 2},
    }
    ckpt = train(cfg)
    assert os.path.exists(ckpt)
    model, tok, _ = load_checkpoint(ckpt)
    s = load_sample(files[0])
    pred = predict_sample(model, tok, s, torch.device("cpu"))
    assert pred.shape == s["dose"].shape and np.isfinite(pred).all() and (pred[~s["body"]] == 0).all()
