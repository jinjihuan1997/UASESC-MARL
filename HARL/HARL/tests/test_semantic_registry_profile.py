import numpy as np
import pytest

from harl.envs.uav_escs.semantic_models.semantic_registry import SemanticModeLibrary


def test_profile_rejects_reordered_mode_ids(tmp_path):
    path = tmp_path / "reordered.npz"
    np.savez(path, snr_grid_db=[0., 10.], q_hat_mean=[[20., 30.], [25., 35.]],
             bar_ls_main_mean=[[1., 1.], [2., 2.]], mode_ids=["b", "a"])
    with pytest.raises(ValueError, match="mode_ids"):
        SemanticModeLibrary("test", [{"id": "a"}, {"id": "b"}], profile_path=str(path))


def test_offline_profile_interpolates_quality_and_load(tmp_path):
    profile_path = tmp_path / "profile.npz"
    np.savez_compressed(
        profile_path,
        schema_version="harl_semantic_profile_v1",
        semantic_model_set="profile_test",
        dataset_name="toy",
        quality_metric="psnr",
        snr_grid_db=np.asarray([0.0, 10.0, 20.0]),
        q_hat_mean=np.asarray([[20.0, 30.0, 40.0], [25.0, 35.0, 45.0]]),
        bar_ls_main_mean=np.asarray([[100.0, 200.0, 300.0], [120.0, 220.0, 320.0]]),
        avg_kept_real_symbols_mean=np.asarray([[200.0, 400.0, 600.0], [240.0, 440.0, 640.0]]),
    )
    modes = [
        {"id": "m0", "trained_snr_db": 0.0, "mu_comm": 1.0e-4},
        {"id": "m1", "trained_snr_db": 10.0, "mu_comm": 1.0e-3},
    ]
    library = SemanticModeLibrary("profile_test", modes, profile_path=str(profile_path))
    tables = library.predict_mode_tables(
        gamma_bh=np.asarray([10.0]),
        psi_mean=np.zeros((1, 2), dtype=float),
        h_s=32,
        w_s=32,
        side_info_bits=8.0,
    )

    side_uses = 8.0 / np.log2(1.0 + 10.0)
    np.testing.assert_allclose(tables["Q_hat_rec"][0, :, 0], [30.0, 30.0])
    np.testing.assert_allclose(tables["Q_hat_rec"][0, :, 1], [35.0, 35.0])
    np.testing.assert_allclose(tables["L_z"][0, :, 0], [200.0, 200.0])
    np.testing.assert_allclose(tables["n_z"][0, :, 1], [440.0, 440.0])
    np.testing.assert_allclose(tables["Lambda_sem"][0, :, 0], [200.0 + side_uses, 200.0 + side_uses])

    diag = library.diagnostic()
    assert diag["semantic_profile_enabled"] is True
    assert diag["semantic_profile_quality_metric"] == "psnr"
    assert diag["semantic_profile_dataset_name"] == "toy"


def test_offline_profile_rejects_wrong_mode_count(tmp_path):
    profile_path = tmp_path / "bad_profile.npz"
    np.savez_compressed(
        profile_path,
        semantic_model_set="profile_test",
        snr_grid_db=np.asarray([0.0, 10.0]),
        q_hat_mean=np.asarray([[20.0, 30.0]]),
        bar_ls_main_mean=np.asarray([[100.0, 200.0]]),
    )
    modes = [{"id": "m0"}, {"id": "m1"}]
    try:
        SemanticModeLibrary("profile_test", modes, profile_path=str(profile_path))
    except ValueError as exc:
        assert "q_hat_mean shape" in str(exc)
    else:
        raise AssertionError("Expected invalid profile shape to be rejected")
