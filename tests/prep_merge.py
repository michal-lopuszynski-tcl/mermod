import copy
import logging
import os
import pathlib

import helpers
import torch

import mermod

logger = logging.getLogger(__name__)


def setup_logging():
    fmt = "%(asctime)s %(levelname)s %(name)s  %(message)s"

    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Herer you put modules that you want more verbose logging

    for module_name in [__name__, "mermod"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def merge(config, safetensors):
    config_merge = copy.deepcopy(config)

    sd_fname_template = "tmp_%02d.pt"
    sd = helpers.gen_state_dict(config["sd_base_path"])
    sd_base_path = sd_fname_template % 0
    config_merge["sd_base_path"] = sd_base_path
    paths_to_del = [sd_base_path]
    torch.save(sd, config_merge["sd_base_path"])

    for i, (k, seed) in enumerate(config["sd_merged_paths"].items(), start=1):
        sd = helpers.gen_state_dict(seed)
        sd_fname = sd_fname_template % i
        config_merge["sd_merged_paths"][k] = sd_fname
        torch.save(sd, sd_fname)
        paths_to_del.append(sd_fname)

    if safetensors:
        p = pathlib.Path(config_merge["sd_output_path"])
        new_p = pathlib.Path("tests/data_safetensors/") / (p.stem + ".safetensors")
        config_merge["sd_output_path"] = new_p

    try:
        _, seed_dict, timing_dict = mermod.merge(**config_merge)
    finally:
        for p in paths_to_del:
            pathlib.Path(p).unlink()

    print(seed_dict)

    out_path = config_merge["sd_output_path"]
    cmd = f"chmod ugo-w {out_path}"
    os.system(cmd)


if __name__ == "__main__":
    IGNORE_EXCEPTIONS = False

    setup_logging()
    configs = [
        helpers.CONFIG_ABS_DIFF_TIES0,
        helpers.CONFIG_ABS_DIFF_TIES1,
        helpers.CONFIG_JOINT_ABS_DIFF_TIES0,
        helpers.CONFIG_JOINT_ABS_DIFF_TIES1,
        helpers.CONFIG_DARE_TIES1_CPU,
        helpers.CONFIG_DARE_TIES1_CUDA,
        helpers.CONFIG_DARE_DISJOINT_TIES1_CPU,
        helpers.CONFIG_DARE_DISJOINT_TIES1_CUDA,
    ]

    for c in configs:
        try:
            merge(c, safetensors=False)
            # Perhaps this will be useful at some point
            # merge(c, safetensors=True)
        except Exception as e:
            if IGNORE_EXCEPTIONS:
                logger.error(f"Exception {e=}")
            else:
                raise e
