import copy
import logging
import pathlib

import helpers
import pytest
import torch

import mermod
import mermod.io

logger = logging.getLogger(__name__)


def check_config(config, device, sd_path: pathlib.Path, data_format):
    config_merge = copy.deepcopy(config)
    if data_format == "pt":
        config_merge["sd_output_path"] = sd_path / "tmp_XX.pt"
        sd_fname_template = "tmp_%02d.pt"
    elif data_format == "safetensors":
        config_merge["sd_output_path"] = sd_path / "tmp_XX.safetensors"
        sd_fname_template = "tmp_%02d.safetensors"

    config_merge["merge_device"] = device

    sd = helpers.gen_state_dict(config["sd_base_path"])
    sd_fname = sd_path / (sd_fname_template % 0)
    config_merge["sd_base_path"] = sd_fname
    mermod.io.save_partial_sd(sd, sd_fname)

    for i, (k, seed) in enumerate(config["sd_merged_paths"].items(), start=1):
        sd = helpers.gen_state_dict(seed)
        sd_fname = sd_path / (sd_fname_template % i)
        config_merge["sd_merged_paths"][k] = sd_fname
        mermod.io.save_partial_sd(sd, sd_fname)

    mermod.merge(**config_merge)
    sd, _ = mermod.io.load_partial_sd(config_merge["sd_output_path"], None, device)
    sd_exp, _ = mermod.io.load_partial_sd(config["sd_output_path"], None, device)
    assert set(sd.keys()) == set(sd_exp.keys())

    for k, wk_exp in sd_exp.items():
        wk = sd[k]
        torch.testing.assert_close(wk.cpu(), wk_exp.cpu())


def test_abs_diff_ties0_cpu_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES0, "cpu", tmp_path, "pt")


def test_abs_diff_ties0_cpu_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES0, "cpu", tmp_path, "safetensors")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_abs_diff_ties0_cuda_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES0, "cuda", tmp_path, "pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_abs_diff_ties0_cuda_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES0, "cuda", tmp_path, "safetensors")


def test_abs_diff_ties1_cpu_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES1, "cpu", tmp_path, "pt")


def test_abs_diff_ties1_cpu_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES1, "cpu", tmp_path, "safetensors")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_abs_diff_ties1_cuda_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES1, "cuda", tmp_path, "pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_abs_diff_ties1_cuda_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_ABS_DIFF_TIES1, "cuda", tmp_path, "safetensors")


def test_joint_abs_diff_ties0_cpu_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES0, "cpu", tmp_path, "pt")


def test_joint_abs_diff_ties0_cpu_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES0, "cpu", tmp_path, "safetensors")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_joint_abs_diff_ties0_cuda_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES0, "cuda", tmp_path, "pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_joint_abs_diff_ties0_cuda_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES0, "cuda", tmp_path, "safetensors")


def test_joint_abs_diff_ties1_cpu_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES1, "cpu", tmp_path, "pt")


def test_joint_abs_diff_ties1_cpu_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES1, "cpu", tmp_path, "safetensors")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_joint_abs_diff_ties1_cuda_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES1, "cuda", tmp_path, "pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_joint_abs_diff_ties1_cuda_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_JOINT_ABS_DIFF_TIES1, "cuda", tmp_path, "safetensors")


def test_dare_ties1_cpu_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_TIES1_CPU, "cpu", tmp_path, "pt")


def test_dare_ties1_cpu_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_TIES1_CPU, "cpu", tmp_path, "safetensors")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_dare_ties1_cuda_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_TIES1_CUDA, "cuda", tmp_path, "pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_dare_ties1_cuda_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_TIES1_CUDA, "cuda", tmp_path, "safetensors")


def test_dare_disjoint_ties1_cpu_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_DISJOINT_TIES1_CPU, "cpu", tmp_path, "pt")


def test_dare_disjoint_ties1_cpu_safetensors(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_DISJOINT_TIES1_CPU, "cpu", tmp_path, "safetensors")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_dare_disjoint_ties1_cuda_pt(tmp_path: pathlib.Path):
    check_config(helpers.CONFIG_DARE_DISJOINT_TIES1_CUDA, "cuda", tmp_path, "pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_dare_disjoint_ties1_cuda_safetensors(tmp_path: pathlib.Path):
    check_config(
        helpers.CONFIG_DARE_DISJOINT_TIES1_CUDA, "cuda", tmp_path, "safetensors"
    )
