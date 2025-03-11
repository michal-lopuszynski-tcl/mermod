import collections
import datetime
import functools
import gc
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def setup_logging():
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Herer you put modules that you want more verbose logging

    for module_name in [__name__, "evaluate"]:
        logging.getLogger(module_name).setLevel(logging.INFO)


def mkdir_tmp():
    output_dir_str = datetime.datetime.now().strftime("./tmp_sd_%Y%m%d_%H%M%S_%f")[:24]
    output_dir = pathlib.Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Created temporary directory {output_dir}")
    return output_dir


def partition_to_batches(lst: list[Any], batch_size: int, reverse: bool) -> list[Any]:
    batches = [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]
    if reverse:
        return [batch[::-1] for batch in batches][::-1]
    return batches


def find_duplicates(batches):
    res_set = set()
    duplicates = []

    for b in batches:
        for bi in b:
            if bi in res_set:
                duplicates.append(bi)
            else:
                res_set.add(bi)
    return duplicates


def check_if_unique(batches):
    duplicates = find_duplicates(batches)
    assert len(duplicates) == 0, f"Duplicate weights found {duplicates}"


def prepare_batches(*, weight_names, weight_batch_size, weight_batches_custom, reverse):

    if weight_batches_custom:
        weights_to_skip = set()
        for b in weight_batches_custom:
            weights_to_skip.update(b)
        weight_names_no_custom = [w for w in weight_names if w not in weights_to_skip]
        res = weight_batches_custom + partition_to_batches(
            weight_names_no_custom, weight_batch_size, reverse
        )
    else:
        res = partition_to_batches(weight_names, weight_batch_size, reverse)
    check_if_unique(res)
    return res


def delete_sd(sd):
    for k in sd:
        v = sd[k]
        sd[k] = None
        del v


def get_weight_names(sd_path):
    start = time.perf_counter()
    sd = torch.load(sd_path, weights_only=True)
    names = [k for k in sd.keys()]
    delete_sd(sd)
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


def merge_partial_sds(output_path, partial_sd_paths, device):
    start = time.perf_counter()
    sd = torch.load(partial_sd_paths[0], weights_only=True, map_location=device)
    for sdp in partial_sd_paths[1:]:
        sd.update(torch.load(sdp, weights_only=True))
    merging_time = time.perf_counter() - start

    n_sds = len(partial_sd_paths)
    n_weights = len(sd)

    torch.save(sd, output_path)
    logger.info(f"t = {merging_time:5.2f} s - merging {n_sds=} with {n_weights=}")

    return sd, merging_time


# def get_cpu_reserved_memory_gb_psutil():
#     import psutil
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info().rss / float(2**30)
#     return mem


def get_cpu_reserved_memory_gb():
    # Get current process ID
    pid = os.getpid()

    # Read memory info from /proc/[pid]/status
    with open(f"/proc/{pid}/status", "r") as f:
        for line in f:
            if "VmRSS:" in line:
                # Extract the memory value (in kB)
                memory_gb = int(line.split()[1])
                # Convert to MB
                memory_mb = memory_gb / 1024 / 1024
                return memory_mb

    return None


def get_gpu_reserved_memory_gb() -> float:
    if torch.cuda.is_available():
        mem = sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )
        return mem / (1024.0**3)
    else:
        return 0.0


def log_memory(msg):
    mem_cpu = get_cpu_reserved_memory_gb()
    mem_gpu = get_gpu_reserved_memory_gb()
    logger.info(f"MEM: CPU {mem_cpu:.3f} GB, GPU={mem_gpu:.3f} GB {msg}")


def free_memory(msg: str = "") -> None:
    mem_cpu1 = get_cpu_reserved_memory_gb()
    mem_gpu1 = get_gpu_reserved_memory_gb()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_cpu2 = get_cpu_reserved_memory_gb()
    mem_gpu2 = get_gpu_reserved_memory_gb()
    d_cpu = mem_cpu1 - mem_cpu2
    d_gpu = mem_gpu1 - mem_gpu2
    info = (
        f"MEM: CPU {mem_cpu1:.3f} -> {mem_cpu2:.3f} [freed {d_cpu:.3f}] GB, "
        f" GPU {mem_gpu1:.3f} -> {mem_gpu2:.3f} [freed {d_gpu:.3f}] GB"
    )
    if msg:
        info + f" {msg}"
    logger.info(info)


# MERGING LOGIC


# def get_random_diff(diff: torch.Tensor, sparsity: float, seed, device):
#     gen = torch.Generator()
#     gen.manual_seed(seed)
#     rand_tensor = torch.rand(size=diff.shape, generator=gen)
#     mask = torch.where(rand_tensor > sparsity, 1.0, 0.0)
#     mask = mask.to(device)
#     return diff * mask, mask


def get_tensor_size_mb(w: torch.Tensor) -> float:
    return (w.nelement() * w.element_size()) / 1024**2


def get_rand(*, size, device, gen, seed):
    gen.manual_seed(seed)
    return torch.rand(size=size, generator=gen, device=device)


def get_randperm(*, n, device, gen, seed):
    gen.manual_seed(seed)
    return torch.randperm(n, generator=gen, device=device)


def mask_random(*, t: torch.Tensor, sparsity: float, device, gen, seed):
    rand_tensor = get_rand(size=t.shape, device=device, gen=gen, seed=seed)
    mask = torch.where(rand_tensor > sparsity, 1.0, 0.0)
    return t * mask, mask


def create_random_interleaved_masks(k: int, shape, gen, seed, device):
    # Create a tensor of indices
    total_elements = torch.prod(torch.tensor(shape)).item()
    log_memory(f"Creating indices - num elements {total_elements/1.0e6} M")
    indices = get_randperm(n=total_elements, device=device, gen=gen, seed=seed)
    logger.info(f"Created inidices - tensor size {get_tensor_size_mb(indices)} MB")

    # Calculate elements per mask (ensuring equal distribution)
    elements_per_mask = total_elements // k

    # Create empty masks
    masks = torch.zeros((k,) + shape, device=device)

    # Fill each mask with 1s in its designated positions
    for i in range(k):
        start_idx = i * elements_per_mask
        end_idx = (i + 1) * elements_per_mask if i < k - 1 else total_elements

        # Get the random positions for this mask
        mask_indices = indices[start_idx:end_idx]

        # Convert linear indices to subscript indices
        if len(shape) == 1:
            masks[i].view(-1)[mask_indices] = 1.0
        else:
            row_indices = mask_indices // shape[1]
            col_indices = mask_indices % shape[1]
            masks[i][row_indices, col_indices] = 1.0

    return masks


def get_pruned_diff(
    diff: torch.Tensor,
    sparsity: float,
    per_row: bool = False,
    per_column: bool = False,
):
    if per_row and len(diff.shape) == 2:
        masks = []
        for row_id in range(diff.shape[0]):
            _, row_mask = get_pruned_diff(diff[row_id], sparsity=sparsity)
            masks.append(row_mask)

        mask = torch.stack(masks, dim=0)
        return mask * diff, mask
    if per_column and len(diff.shape) == 2:
        masks = []
        for column_id in range(diff.shape[1]):
            _, column_mask = get_pruned_diff(diff[:, column_id], sparsity=sparsity)
            masks.append(column_mask)

        mask = torch.stack(masks, dim=1)
        return mask * diff, mask

    abs_diff = torch.abs(diff)
    original_shape = diff.shape
    _, indices = torch.topk(abs_diff.flatten(), k=int((1 - sparsity) * diff.numel()))
    mask = torch.zeros_like(abs_diff.flatten())
    mask[indices] = 1.0
    mask = mask.reshape(original_shape)
    return diff * mask, mask


def get_joint_pruned_diff(
    task_vectors: dict[str, torch.Tensor],
    sparsity: float,
):
    abs_diffs = torch.stack([torch.abs(t) for t in task_vectors.values()], dim=0)
    device = abs_diffs.device
    original_shape = abs_diffs.shape
    n_abs_diffs = abs_diffs.numel()
    _, indices = torch.topk(abs_diffs.flatten(), k=int((1 - sparsity) * n_abs_diffs))
    del abs_diffs
    masks = torch.zeros(n_abs_diffs, device=device)
    masks[indices] = 1.0
    masks = masks.reshape(original_shape)
    masks_list = torch.unbind(masks, dim=0)

    for_iter = zip(task_vectors.keys(), masks_list, task_vectors.values())

    for task_name, mask, task_vector in for_iter:
        task_vectors[task_name] = mask * task_vector
        del task_vector

    return task_vectors, masks_list


# def merge_to_base_sd_old(
#     *,
#     base_sd,
#     merged_sds,
#     method: str = "dare",
#     lambda_param: Optional[float] = 1.0,
#     sparsity: float = 0.9,
#     use_ties: bool = False,
#     seed_dict: Optional[dict[str, int]] = None,
#     device: torch.device,
# ):
#     logger.info("OLD VERSION")
#     # TODO Matter only for abs_diff, either remove them totally or put them in config
#     # per_row = False
#     # per_column = False

#     start = time.perf_counter()

#     gen_for_seeds = torch.Generator()

#     new_seed_dict = {}
#     n = len(base_sd)

#     for k, weight_name in enumerate(base_sd, start=1):

#         # BUILD TASK VECTORS
#         # logger.info(f"MEM 2.1 {get_cur_mem_gb()=:.2f}")
#         task_vectors = {
#             sd_name: (sd[weight_name] - base_sd[weight_name])
#             for sd_name, sd in merged_sds.items()
#         }
#         # logger.info(f"MEM 2.2 {get_cur_mem_gb()=:.2f}")
#         weight_size = get_weight_size_mb(base_sd[weight_name])
#         msg = f"Started weight {k} of {n}: {weight_name} [{weight_size:.1f} MB]"
#         if torch.stack([t.abs() for t in task_vectors.values()]).mean() == 0:
#             logger.info(f"{msg} - untrained parameters, skipping")
#             continue
#         else:
#             logger.info(f"{msg}")
#         logger.info(f"MEM GPU {get_gpu_reserved_memory_gb()} GB")

#         # BUILD TRIMMED_TASK_VECTORS AND MASKS

#         trimmed_task_vectors = []
#         masks = []

#         if method.lower() == "dare":
#             for task_vector_name, task_vector in task_vectors.items():
#                 seed_id = f"{weight_name}@{task_vector_name}"
#                 if seed_dict is not None:
#                     seed = seed_dict[seed_id]
#                 else:
#                     seed = gen_for_seeds.seed()
#                     new_seed_dict[seed_id] = seed
#                 pruned_diff, mask = get_random_diff(task_vector, sparsity,
#                                           seed, device)
#                 trimmed_task_vectors.append(pruned_diff)
#                 masks.append(mask)

#         # elif method.lower() == "dare_disjoint":
#         #     initial_masks = create_random_interleaved_masks(
#         #         len(task_vectors), task_vectors[0].shape
#         #     )
#         #     masks = []
#         #     for initial_mask, diff in zip(initial_masks, task_vectors):
#         #         final_mask = torch.where(
#         #             torch.rand(size=initial_mask.shape) > sparsity,
#                       initial_mask, 0.0
#         #         )
#         #         masks.append(final_mask)
#         #         trimmed_task_vectors.append(final_mask * diff)

#         # elif method.lower() == "abs_diff":
#         #     for k, diff in enumerate(task_vectors):
#         #         pruned_diff, mask = get_pruned_diff(
#         #             diff, sparsity, per_row=per_row, per_column=per_column
#         #         )
#         #         trimmed_task_vectors.append(pruned_diff)
#         #         masks.append(mask)

#         # elif method.lower() == "joint_abs_diff":
#         #     trimmed_task_vectors, masks = get_joint_pruned_diff(
#         #         task_vectors=task_vectors, sparsity=sparsity
#         #     )
#         # else:
#         #     raise ValueError(f'Unknown merge method - "{method}"')

#         # LOG STATISTICS
#         # for k, mask, pruned_diff, diff in zip(
#         #     range(len(masks)), masks, trimmed_task_vectors, task_vectors
#         # ):
#         #     selected_values_abs_mean = (
#         #         pruned_diff.abs().sum() / (mask == 1.0).sum()
#         #     ).item()
#         #     discarded_values_abs_mean = (
#         #         ((1.0 - mask) * diff.abs()).sum() / (mask == 0.0).sum()
#         #     ).item()
#         #     logger.info(f"Diff index: {k}, mask mean: {mask.mean().item()}")
#         #     logger.info(
#         #         f"Diff index: {k}, abs +: {selected_values_abs_mean}, "
#         #         f"abs -: {discarded_values_abs_mean}"
#         #     )

#         mask_intersection = masks[0]
#         for tensor in masks[1:]:
#             mask_intersection *= tensor
#         # logger.info(
#         #     f"Param: {name}, mask sum: {mask_sum.mean().item()}, mask intersection:"
#         #     f" {mask_intersection.mean().item()}"
#         # )

#         # APPLY TIES ALIGNMENT
#         # sum_values = torch.sum(torch.stack(trimmed_task_vectors), dim=0)
#         sum_values = torch.zeros_like(trimmed_task_vectors[0], dtype=torch.float32)
#         for tv in trimmed_task_vectors:
#             sum_values += tv
#         sign = torch.sign(sum_values)
#         sum_values = None
#         aligned_trimmed_task_vectors = []
#         for diff in trimmed_task_vectors:
#             aligned_diff = (
#                 torch.where(torch.sign(diff) == sign, diff, 0.0) if use_ties else diff
#             )
#             aligned_trimmed_task_vectors.append(aligned_diff)

#         # CREATE MERGED PARAM VALUE

#         if lambda_param is None:
#             mask_sum = functools.reduce(torch.max, masks)
#             current_lambda_param = 1.0 / mask_sum.mean()
#             logger.info(f"Current lambda: {current_lambda_param}")
#         else:
#             current_lambda_param = lambda_param

#         for diff in aligned_trimmed_task_vectors:
#             base_sd[weight_name] += current_lambda_param * diff
#         # logger.info(f"MEM 2.3 {get_cur_mem_gb()=:.2f}")
#         del task_vectors
#         del aligned_trimmed_task_vectors
#         del diff
#         del masks
#         gc.collect()
#         free_gpu_reserved_memory()
#         msg = f"Finished weight {k} of {n}: {weight_name} [{weight_size:.1f} MB]"
#         # logger.info(f"MEM 2.4 {get_cur_mem_gb()=:.2f}")
#     time_merging = time.perf_counter() - start
#     logger.info(f"t = {time_merging:5.2f} s - merging {len(base_sd)} layers")

#     if seed_dict is None:
#         return new_seed_dict, time_merging
#     else:
#         return seed_dict, time_merging


def get_first_value(d):
    return next(iter(d.values()))


def apply_ties_in_place(task_vectors):
    sum_values = torch.zeros_like(get_first_value(task_vectors), dtype=torch.float32)
    for tv in task_vectors.values():
        sum_values += tv
    sign = torch.sign(sum_values)
    del sum_values

    for task_name, task_vector in task_vectors.items():
        aligned_task_vector = torch.where(
            torch.sign(task_vector) == sign, task_vector, 0.0
        )
        task_vectors[task_name] = aligned_task_vector
        del task_vector


def all_tasks_vectors_zero(task_vectors):
    for task_vector in task_vectors.values():
        if torch.max(torch.abs(task_vector)) > 1.0e-6:
            return False
    return True


def merge_to_base_sd(
    *,
    base_sd,
    merged_sds,
    method: str = "dare",
    lambda_param: Optional[float] = 1.0,
    sparsity: float = 0.9,
    use_ties: bool = False,
    seed_dict: Optional[dict[str, int]] = None,
    device: torch.device,
):
    # TODO Matter only for abs_diff, either remove them totally or put them in config
    per_row = False
    per_column = False

    start = time.perf_counter()
    if str(device).startswith("cuda"):
        sfx = "_cuda"
    else:
        sfx = "_cpu"

    gen_for_seeds = torch.Generator()
    gen_for_tensors = torch.Generator(device=device)

    new_seed_dict = {}
    n = len(base_sd)

    for k, weight_name in enumerate(base_sd, start=1):
        free_memory(f"{weight_name} - start")

        # BUILD TASK VECTORS
        task_vectors = {
            sd_name: (sd[weight_name] - base_sd[weight_name])
            for sd_name, sd in merged_sds.items()
        }

        log_memory(f"{weight_name} - task vectors")

        msg = f"Weight {k} of {n}: {weight_name}"
        if all_tasks_vectors_zero(task_vectors):
            logger.info(f"{msg} untrained parameters, skipping")
            continue
        else:
            weight_size = get_tensor_size_mb(base_sd[weight_name])
            logger.info(f"{msg} size={weight_size:.1f} MB")

        # CREATE MASKS AND TRIM TASK VECTOS

        masks = []

        if method.lower() == "dare":
            for task_name, task_vector in task_vectors.items():
                seed_id = f"{weight_name}@{task_name}:{sfx}"
                if seed_dict is not None:
                    seed = seed_dict[seed_id]
                else:
                    assert seed_id not in new_seed_dict
                    seed = gen_for_seeds.seed()
                    new_seed_dict[seed_id] = seed
                pruned_task_vector, mask = mask_random(
                    t=task_vector,
                    sparsity=sparsity,
                    seed=seed,
                    gen=gen_for_tensors,
                    device=device,
                )
                task_vectors[task_name] = pruned_task_vector
                masks.append(mask)

        elif method.lower() == "dare_disjoint":
            log_memory("dare disjoint start")
            seed_id = f"{weight_name}@PERMUTATION:{sfx}"
            if seed_dict is not None:
                seed = seed_dict[seed_id]
            else:
                seed = gen_for_seeds.seed()
                new_seed_dict[seed_id] = seed
            shape = get_first_value(task_vectors).shape
            initial_masks = create_random_interleaved_masks(
                k=len(task_vectors),
                shape=shape,
                gen=gen_for_tensors,
                seed=seed,
                device=device,
            )
            task_iter = zip(task_vectors.keys(), task_vectors.values(), initial_masks)
            for task_name, task_vector, mask0 in task_iter:
                seed_id = f"{weight_name}@{task_name}:{sfx}"
                if seed_dict is not None:
                    seed = seed_dict[seed_id]
                else:
                    seed = gen_for_seeds.seed()
                    new_seed_dict[seed_id] = seed
                rand_mask = get_rand(
                    size=mask0.shape, device=device, gen=gen_for_tensors, seed=seed
                )
                mask1 = torch.where(rand_mask > sparsity, mask0, 0.0)
                task_vectors[task_name] = mask1 * task_vector
                masks.append(mask1)

        elif method.lower() == "abs_diff":
            for task_name, task_vector in task_vectors.items():
                pruned_task_vector, mask = get_pruned_diff(
                    task_vector, sparsity, per_row=per_row, per_column=per_column
                )
                task_vectors[task_name] = pruned_task_vector
                masks.append(mask)
                del task_vector
        elif method.lower() == "joint_abs_diff":
            task_vectors, masks = get_joint_pruned_diff(
                task_vectors=task_vectors, sparsity=sparsity
            )
        else:
            raise ValueError(f'Unknown merge method - "{method}"')

        # LOG STATISTICS
        # for k, mask, pruned_diff, diff in zip(
        #     range(len(masks)), masks, trimmed_task_vectors, task_vectors
        # ):
        #     selected_values_abs_mean = (
        #         pruned_diff.abs().sum() / (mask == 1.0).sum()
        #     ).item()
        #     discarded_values_abs_mean = (
        #         ((1.0 - mask) * diff.abs()).sum() / (mask == 0.0).sum()
        #     ).item()
        #     logger.info(f"Diff index: {k}, mask mean: {mask.mean().item()}")
        #     logger.info(
        #         f"Diff index: {k}, abs +: {selected_values_abs_mean}, "
        #         f"abs -: {discarded_values_abs_mean}"
        #     )

        # mask_intersection = masks[0]
        # for tensor in masks[1:]:
        #     mask_intersection *= tensor
        # logger.info(
        #     f"Param: {name}, mask sum: {mask_sum.mean().item()}, mask intersection:"
        #     f" {mask_intersection.mean().item()}"
        # )

        # APPLY TIES ALIGNMENT

        if use_ties:
            logger.info("Using ties")
            apply_ties_in_place(task_vectors)
        else:
            logger.info("Not using ties")

        # CREATE MERGED PARAM VALUE

        if lambda_param is None:
            mask_sum = functools.reduce(torch.max, masks)
            current_lambda_param = 1.0 / mask_sum.mean()
            logger.info(f"Current lambda: {current_lambda_param}")
        else:
            current_lambda_param = lambda_param

        for task_vector in task_vectors.values():
            base_sd[weight_name] += current_lambda_param * task_vector

        del task_vectors
        del masks
        gc.collect()
        free_memory()
        msg = f"Finished weight {k} of {n}: {weight_name} [{weight_size:.1f} MB]"

    time_merging = time.perf_counter() - start
    logger.info(f"t = {time_merging:5.2f} s - merging {len(base_sd)} layers")

    if seed_dict is None:
        return new_seed_dict, time_merging
    else:
        return seed_dict, time_merging


def log_sd(sd_name, sd):
    sd_size = 0
    for k, v in sd.items():
        sd_size += sys.getsizeof(v.untyped_storage())
    logger.info(f"MEM {sd_name}: len={len(sd)} size={sd_size/1024**3:.2f} GB")


def _merge_one_batch(
    *,
    sd_base_path,
    sd_merged_paths,
    names_batch,
    method,
    sparsity,
    use_ties,
    lambda_param,
    seed_dict,
    seed_dict_new,
    sd_out_path,
    device,
):
    tot_t_compute, tot_t_io = 0.0, 0.0
    base_partial_sd, t_io = load_partial_sd(sd_base_path, names_batch, device)
    tot_t_io += t_io

    merged_partial_sds = {}
    for sd_name, sd_path in sd_merged_paths.items():
        sd, t_io = load_partial_sd(sd_path, names_batch, device)
        tot_t_io += t_io
        merged_partial_sds[sd_name] = sd
        log_sd(sd_name, sd)

    with torch.no_grad():
        partial_seed_dict, t_compute = merge_to_base_sd(
            base_sd=base_partial_sd,
            merged_sds=merged_partial_sds,
            method=method,
            sparsity=sparsity,
            lambda_param=lambda_param,
            use_ties=use_ties,
            seed_dict=seed_dict,
            device=device,
        )
    tot_t_compute += t_compute

    if seed_dict is None:
        seed_dict_new.update(partial_seed_dict)

    tot_t_io += save_partial_sd(base_partial_sd, sd_out_path)

    for k in merged_partial_sds:
        v = merged_partial_sds[k]
        merged_partial_sds[k] = None
        delete_sd(v)

    del merged_partial_sds
    delete_sd(base_partial_sd)
    del base_partial_sd

    gc.collect()

    return tot_t_compute, tot_t_io


def _merge(
    *,
    sd_base_path,
    sd_merged_paths,
    method,
    sparsity,
    use_ties,
    lambda_param,
    weight_batch_size,
    weight_batches_custom,
    seed_dict,
    merging_tmp_dir,
    device,
):
    tot_t_io, tot_t_compute = 0.0, 0.0
    logger.info(f"{sd_base_path=}")
    logger.info(f"{sd_merged_paths=}")

    weight_names, t_io = get_weight_names(sd_base_path)
    tot_t_io += t_io
    weights_names_batches = prepare_batches(
        weight_names=weight_names,
        weight_batch_size=weight_batch_size,
        weight_batches_custom=weight_batches_custom,
        reverse=True,
    )

    n_weights = len(weight_names)
    n_batches = len(weights_names_batches)
    partial_paths = []
    logger.info(f"{n_weights=} {n_batches=} {weight_batch_size=}")

    seed_dict_new = {}

    for i, names_batch in enumerate(weights_names_batches, start=1):
        mem1 = get_cpu_reserved_memory_gb()
        mem1_gpu = get_gpu_reserved_memory_gb()

        sd_out_path = merging_tmp_dir / f"sd-{i:05d}-{n_batches:05d}.pth"
        t_compute, t_io = _merge_one_batch(
            sd_base_path=sd_base_path,
            sd_merged_paths=sd_merged_paths,
            sd_out_path=sd_out_path,
            names_batch=names_batch,
            method=method,
            sparsity=sparsity,
            use_ties=use_ties,
            lambda_param=lambda_param,
            seed_dict=seed_dict,
            seed_dict_new=seed_dict_new,
            device=device,
        )
        tot_t_compute += t_compute
        tot_t_io += t_io
        partial_paths.append(sd_out_path)

        mem2 = get_cpu_reserved_memory_gb()

        free_memory()
        mem2_gpu = get_gpu_reserved_memory_gb()

        logger.info(f"MEM {mem2=:.2f} {mem2_gpu=:.2f}")
        memd = mem1 - mem2
        prefix = "MEM CPU usage during iteration - "
        logger.info(f"{prefix} {mem1:.2f} -> {mem2:.2f} GB [collected {memd:.2f} GB]")
        prefix = "MEM GPU usage during iteration - "
        memd = mem1_gpu - mem2_gpu
        msg = f"{prefix} {mem1_gpu:.2f} -> {mem2_gpu:.2f} GB [collected {memd:.2f} GB]"
        logger.info(msg)

    if seed_dict is None:
        return partial_paths, seed_dict_new, (tot_t_compute, tot_t_io)
    else:
        return partial_paths, seed_dict, (tot_t_compute, tot_t_io)


def merge(
    *,
    sd_base_path,
    sd_merged_paths,
    method,
    sparsity,
    use_ties,
    lambda_param,
    weight_batch_size,
    weight_batches_custom,
    sd_output_path,
    seed_dict,
    merge_device,
):
    logger.info(f"{merge_device=}")
    start = time.perf_counter()
    tot_t_compute, tot_t_io = 0.0, 0.0

    merging_tmp_dir = mkdir_tmp()
    try:
        partial_paths, seed_dict, (t_compute, t_io) = _merge(
            sd_base_path=sd_base_path,
            sd_merged_paths=sd_merged_paths,
            method=method,
            sparsity=sparsity,
            use_ties=use_ties,
            lambda_param=lambda_param,
            weight_batch_size=weight_batch_size,
            weight_batches_custom=weight_batches_custom,
            merging_tmp_dir=merging_tmp_dir,
            seed_dict=seed_dict,
            device=merge_device,
        )
        tot_t_compute += t_compute
        tot_t_io += t_io

        merge_device = torch.device("cpu")
        sd, t_io = merge_partial_sds(sd_output_path, partial_paths, merge_device)
        tot_t_io += t_io
        log_memory("after merging")
    finally:
        shutil.rmtree(merging_tmp_dir)
    time_full = time.perf_counter() - start
    logger.info(f"Merge time io:      {tot_t_io/60.0:6.2f} min")
    logger.info(f"Merge time compute: {tot_t_compute/60.0:6.2f} min")
    logger.info(f"Merge time tot:     {time_full/60.0:6.2f} min")
    timing_dict = {
        "merge_time_io": tot_t_io,
        "merge_time_compute": tot_t_compute,
        "merge_time_total": time_full,
    }
    return sd, seed_dict, timing_dict
