import datetime
import json
import logging
import pathlib
import random
import os
import string
import time
from typing import Any


import transformers
import mermod
import torch
import lm_eval
import tabulate

logger = logging.getLogger()


MODEL_NAME = "Qwen/Qwen2.5-1.5B"


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


def get_random_string(n):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))


def get_timestamp_for_fname() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)

    # Round to nearest hundredth of a second
    hundredths = round(now.microsecond / 10000)

    # Handle the case where rounding results in 100
    if hundredths == 100:
        now = now + datetime.timedelta(seconds=1)
        hundredths = 0

    now_str = f"{now:%Y-%m-%d-%H%M-%S}{hundredths:02d}"
    return now_str


def get_random_str(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def get_tmp_suffix():
    return f"{get_timestamp_for_fname()}_{get_random_str(6)}"


def calc_lm_eval_metrics(
    model: torch.nn.Module,
    tasks: list[str],
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
    limit: float,
) -> tuple[dict[str, Any], str]:

    lm_eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=model, tokenizer=tokenizer, device=device
    )

    results = lm_eval.evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        device=device,
        confirm_run_unsafe_code=True,
        limit=limit,
    )
    results_str = lm_eval.utils.make_table(results)

    # Make results JSON-serializeable
    results["config"]["device"] = str(results["config"]["device"])
    results["config"]["model_dtype"] = str(results["config"]["device"])

    return results, results_str


def merge_and_evaluate_dbg(model_name, merge_config):
    return {
        "evaluation": {
            "mbpp_pass_at_1": 0.5,
            "ceval-valid_acc": 0.3,
        }
    }


def merge_and_evaluate(
    merge_config,
    model_name,
):
    pathlib.Path(merge_config["sd_output_path"]).mkdir(parents=True, exist_ok=True)
    ceval_limit = None
    mbpp_limit = 0.5
    device = torch.device("cuda")
    start = time.perf_counter()
    seed_dict, _ = mermod.merge(**merge_config)
    merge_time = time.perf_counter() - start

    model_config = transformers.AutoConfig.from_pretrained(model_name)
    config_path = pathlib.Path(merge_config["sd_output_path"]) / "config.json"
    with open(config_path, "wt") as f:
        f.write(model_config.to_json_string())

    model = transformers.AutoModelForCausalLM.from_pretrained(
        merge_config["sd_output_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    start = time.perf_counter()
    res_mbpp, _ = calc_lm_eval_metrics(
        model=model,
        tokenizer=tokenizer,
        tasks=["mbpp"],
        limit=mbpp_limit,
        device=device,
    )
    mbpp_time = time.perf_counter() - start
    mbpp_pass_at_1 = float(res_mbpp["results"]["mbpp"]["pass_at_1,none"])
    logger.info(f"{mbpp_pass_at_1=}")

    start = time.perf_counter()
    res_ceval, _ = calc_lm_eval_metrics(
        model=model,
        tokenizer=tokenizer,
        tasks=["ceval-valid"],
        limit=ceval_limit,
        device=device,
    )
    ceval_time = time.perf_counter() - start
    ceval_valid_acc = float(res_ceval["results"]["ceval-valid"]["acc,none"])
    logger.info(f"{ceval_valid_acc=}")

    merge_config["seed_dict"] = seed_dict
    merge_id = pathlib.Path(merge_config["sd_output_path"]).stem
    result = {
        "merge_id": merge_id,
        "model_name": model_name,
        "merge_config": merge_config,
        "merge_time": merge_time,
        "evaluation": {
            "mbpp_limit": mbpp_limit,
            "mbpp_pass_at_1": mbpp_pass_at_1,
            "mbpp_time": mbpp_time,
            "ceval_limit": ceval_limit,
            "ceval-valid_acc": ceval_valid_acc,
            "ceval-valid_time": ceval_time,
        },
        "mermod_version": mermod.__version__,
    }
    return result


def main():
    with open("configs/configs.json") as f:
        configs = [json.loads(line) for line in f]

    res = []

    for merge_config in configs:
        res_tmp = merge_and_evaluate(merge_config, MODEL_NAME)
        res.append(
            {
                "model": merge_config["sd_output_path"],
                "mbpp": res_tmp["evaluation"]["mbpp_pass_at_1"],
                "ceval": res_tmp["evaluation"]["ceval-valid_acc"],
            }
        )
    floatfmt = None
    table_str = tabulate.tabulate(
        res,
        headers="keys",
        tablefmt="github",
        # floatfmt=floatfmt)
    )
    print(table_str)


if __name__ == "__main__":
    setup_logging()
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
