import copy
import logging
import os

import helpers
import torch

import mermod


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


def merge(config):
    config_merge = copy.deepcopy(config)
    sd_fname_template = "tmp_%02d.pt"
    sd = helpers.gen_state_dict(config["sd_base_path"])
    config_merge["sd_base_path"] = sd_fname_template % 0
    torch.save(sd, config_merge["sd_base_path"])

    for i, (k, seed) in enumerate(config["sd_merged_paths"].items(), start=1):
        sd = helpers.gen_state_dict(seed)
        sd_fname = sd_fname_template % i
        config_merge["sd_merged_paths"][k] = sd_fname
        torch.save(sd, sd_fname)

    mermod.merge(**config_merge)
    out_path = config_merge["sd_output_path"]
    cmd = f"chmod ugo-w {out_path}"
    os.system(cmd)


if __name__ == "__main__":
    setup_logging()
    merge(helpers.CONFIG_ABS_DIFF_TIES1)
    merge(helpers.CONFIG_ABS_DIFF_TIES0)
