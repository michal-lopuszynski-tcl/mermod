import gc
import logging
import os

import torch

logger = logging.getLogger(__file__)


def delete_sd(sd):
    for k in sd:
        v = sd[k]
        sd[k] = None
        del v


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


def log_memory(logger, msg):
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
