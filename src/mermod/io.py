import collections
import datetime
import gc
import logging
import pathlib
import time

import torch

from . import utils

logger = logging.getLogger(__file__)


def mkdir_tmp():
    output_dir_str = datetime.datetime.now().strftime("./tmp_sd_%Y%m%d_%H%M%S_%f")[:24]
    output_dir = pathlib.Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Created temporary directory {output_dir}")
    return output_dir


def get_weight_names(sd_path):
    start = time.perf_counter()
    sd = torch.load(sd_path, weights_only=True)
    names = [k for k in sd.keys()]
    utils.delete_sd(sd)
    del sd
    gc.collect()
    loading_time = time.perf_counter() - start
    logger.info(f"t = {loading_time:5.2f} s - loading weight names from {sd_path}")
    return names, loading_time


def load_partial_sd(sd_path, weight_names, device):
    start = time.perf_counter()
    sd = torch.load(sd_path, weights_only=True, map_location=device)
    partial_sd = {wn: sd[wn] for wn in weight_names}
    for k in sd:
        if k not in weight_names:
            v = sd[k]  # noqa: F841
            sd[k] = None
            del v
    del sd
    gc.collect()
    loading_time = time.perf_counter() - start
    n = len(weight_names)
    logger.info(f"t = {loading_time:5.2f} s - loading of {n} weights from {sd_path}")
    return collections.OrderedDict(partial_sd), loading_time


def save_partial_sd(sd, sd_path):
    start = time.perf_counter()
    torch.save(sd, sd_path)
    saving_time = time.perf_counter() - start
    n = len(sd)
    logger.info(f"t = {saving_time:5.2f} s - saving of {n} weights to {sd_path}")
    return saving_time
