import collections
import datetime
import gc
import json
import logging
import pathlib
import time
from typing import Optional

import safetensors
import safetensors.torch
import torch

from . import utils

logger = logging.getLogger(__file__)


def mkdir_tmp() -> pathlib.Path:
    output_dir_str = datetime.datetime.now().strftime("./tmp_sd_%Y%m%d_%H%M%S_%f")[:24]
    output_dir = pathlib.Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Created temporary directory {output_dir}")
    return output_dir


def get_sd_type(sd_path: pathlib.Path, should_exist: bool = True) -> str:
    if should_exist and not sd_path.exists():
        raise FileNotFoundError(f"File {sd_path} not found")

    if sd_path.name.endswith(".pt") or sd_path.name.endswith(".pth"):
        return "pt"
    elif sd_path.name.endswith(".safetensors"):
        return "safetensors"
    elif not sd_path.exists() or sd_path.is_dir():
        return "hf"
    else:
        raise ValueError(f"Unrecognized checkpoint path in {sd_path}")


def _get_weight_names_pt(sd_path):
    start = time.perf_counter()
    sd = torch.load(sd_path, weights_only=True)
    names = [k for k in sd.keys()]
    utils.delete_sd(sd)
    del sd
    gc.collect()
    loading_time = time.perf_counter() - start
    logger.info(f"t = {loading_time:5.2f} s - loading weight names from {sd_path}")
    return names, loading_time


def _get_weight_names_safetensors(sd_path: pathlib.Path):
    start = time.perf_counter()
    with safetensors.safe_open(sd_path, framework="pt", device="cpu") as f:
        names = list(f.keys())
    loading_time = time.perf_counter() - start
    return names, loading_time


def _get_weight_names_hf(sd_path):
    start = time.perf_counter()
    with open(sd_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    names = list(index["weight_map"].keys())
    loading_time = time.perf_counter() - start
    return names, loading_time


def get_weight_names(sd_path):
    sd_path = pathlib.Path(sd_path)
    sd_type = get_sd_type(sd_path)
    if sd_type == "pt":
        return _get_weight_names_pt(sd_path)
    elif sd_type == "safetensors":
        return _get_weight_names_safetensors(sd_path)
    elif sd_type == "hf":
        return _get_weight_names_hf(sd_path)
    else:
        raise ValueError(f"Unsupported {sd_type=}")


def _load_partial_sd_pt(sd_path, weight_names: Optional[list[str]], device):
    start = time.perf_counter()

    partial_sd = torch.load(sd_path, weights_only=True, map_location=device)

    # Remove obsolete weights
    if weight_names is not None:
        obsolete_weights = [w for w in partial_sd.keys() if w not in weight_names]
        for k in obsolete_weights:
            del partial_sd[k]

    loading_time = time.perf_counter() - start
    n = len(partial_sd)
    logger.info(f"t = {loading_time:5.2f} s - loading of {n} weights from {sd_path}")
    return partial_sd, loading_time


def _load_partial_sd_safetensors_in_palce(
    partial_sd, sd_path, weight_names, device
) -> None:

    with safetensors.safe_open(sd_path, framework="pytorch", device=device) as f:
        if weight_names is None:
            for wn in f.keys():
                partial_sd[wn] = f.get_tensor(wn)
        else:
            for wn in weight_names:
                if wn in f.keys():
                    partial_sd[wn] = f.get_tensor(wn)
                else:
                    raise ValueError(f"Weight {wn} not found in {sd_path}")


def _load_partial_sd_safetensors(sd_path, weight_names, device):
    start = time.perf_counter()
    sd = collections.OrderedDict()
    _load_partial_sd_safetensors_in_palce(sd, sd_path, weight_names, device)
    loading_time = time.perf_counter() - start
    return sd, loading_time


def _load_partial_sd_hf(sd_path, weight_names, device):
    start = time.perf_counter()
    with open(sd_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_name_to_sd_path = index["weight_map"]
    sd_path_to_weight_names = {}

    for wn in weight_names:
        sd_path = weight_name_to_sd_path[wn]
        if sd_path in sd_path_to_weight_names:
            sd_path_to_weight_names[sd_path].append(wn)
        else:
            sd_path_to_weight_names[sd_path] = [wn]

    sd = collections.OrderedDict()
    for c_sd_path, c_weight_names in sd_path_to_weight_names.items():
        _load_partial_sd_safetensors_in_palce(sd, c_sd_path, c_weight_names, device)
    loading_time = time.perf_counter() - start
    return sd, loading_time


def load_partial_sd(sd_path, weight_names, device):
    sd_path = pathlib.Path(sd_path)
    sd_type = get_sd_type(sd_path)
    if sd_type == "pt":
        return _load_partial_sd_pt(sd_path, weight_names, device)
    elif sd_type == "safetensors":
        return _load_partial_sd_safetensors(sd_path, weight_names, device)
    elif sd_type == "hf":
        return _load_partial_sd_hf(sd_path)
    else:
        raise ValueError(f"Unsupported {sd_type=}")


def _save_partial_sd_pt(sd, sd_path):
    start = time.perf_counter()
    torch.save(sd, sd_path)
    saving_time = time.perf_counter() - start
    n = len(sd)
    msg = f"t = {saving_time:5.2f} s - PyTorch saving of {n} weights to {sd_path}"
    logger.info(msg)
    return saving_time


def _save_partial_sd_safetensors(sd, sd_path):
    start = time.perf_counter()
    safetensors.torch.save_file(sd, sd_path)
    saving_time = time.perf_counter() - start
    n = len(sd)
    msg = f"t = {saving_time:5.2f} s - sfetensors saving of {n} weights to {sd_path}"
    logger.info(msg)
    return saving_time


def save_partial_sd(sd, sd_path):
    sd_path = pathlib.Path(sd_path)
    sd_type = get_sd_type(sd_path, should_exist=False)
    if sd_type == "pt":
        return _save_partial_sd_pt(sd, sd_path)
    elif sd_type == "safetensors":
        return _save_partial_sd_safetensors(sd, sd_path)
    elif sd_type == "hf":
        msg = f"Saving to HF sharded format not yes implemented {sd_path=}"
        raise NotImplementedError(msg)
    else:
        raise ValueError(f"Unsupported {sd_type=}")


def merge_partial_sds(output_path, partial_sd_paths, device):
    start = time.perf_counter()
    output_path = pathlib.Path(output_path)
    sd, _ = load_partial_sd(partial_sd_paths[0], None, device)
    for sd_path in partial_sd_paths[1:]:
        sd_new, _ = load_partial_sd(sd_path, None, device)
        sd.update(sd_new)
    merging_time = time.perf_counter() - start

    n_sds = len(partial_sd_paths)
    n_weights = len(sd)
    if get_sd_type(output_path, should_exist=False) != "hf":
        save_partial_sd(sd, output_path)
    else:
        raise ValueError("Merging state_dicts to hf format not implemented yet")
    logger.info(f"t = {merging_time:5.2f} s - merging {n_sds=} with {n_weights=}")

    return sd, merging_time
