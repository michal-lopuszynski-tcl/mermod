import copy
import logging
import pathlib

import helpers
import pytest
import torch

import mermod

logger = logging.getLogger(__name__)


def check_config(config, device, sd_path: pathlib.Path):
    config_merge = copy.deepcopy(config)
    config_merge["sd_output_path"] = sd_path / "tmp_XX.pt"
    config_merge["merge_device"] = device
    sd_fname_template = "tmp_%02d.pt"
    sd = helpers.gen_state_dict(config["sd_base_path"])
    config_merge["sd_base_path"] = sd_fname_template % 0
    torch.save(sd, config_merge["sd_base_path"])

    for i, (k, seed) in enumerate(config["sd_merged_paths"].items(), start=1):
        sd = helpers.gen_state_dict(seed)
        sd_fname = sd_path / (sd_fname_template % i)
        config_merge["sd_merged_paths"][k] = sd_fname
        torch.save(sd, sd_fname)

    mermod.merge(**config_merge)
    sd = torch.load(config_merge["sd_output_path"], weights_only=True)
    sd_exp = torch.load(config["sd_output_path"])
    assert set(sd.keys()) == set(sd_exp.keys())

    for k, wk_exp in sd_exp.items():
        wk = sd[k]
        torch.testing.assert_close(wk, wk_exp)


def test_dummy_abs_diff_ties0_cpu(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES0, "cpu", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_dummy_abs_diff_ties0_cuda(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES0, "cuda", tmp_path)


def test_dummy_abs_diff_ties1_cpu(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES1, "cpu", tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_dummy_abs_diff_ties1_cuda(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES1, "cuda", tmp_path)
