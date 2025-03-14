import gc
import logging
import os

import torch

try:
    import enlighten
except ImportError:
    enlighten = None

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
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if "VmRSS:" in line:
                    # Extract the memory value (in kB)
                    memory_gb = int(line.split()[1])
                    # Convert to MB
                    memory_mb = memory_gb / 1024 / 1024
                    return memory_mb
        return None
    except Exception:
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


class ProgressBar:

    def __init__(self, total, enabled, desc, units):
        self.enabled = enabled
        self.total = total
        self.desc = desc
        self.units = units
        self.manager = None
        self.counter = None

    def __enter__(self):
        if self.enabled and enlighten is not None:
            self.manager = enlighten.get_manager()
            self.manager = self.manager.__enter__()
            self.counter = self.manager.counter(
                total=self.total, desc=self.desc, units=self.units
            )
            self.counter = self.counter.__enter__()
        return self

    def update(self, incr: int = 1):
        if self.counter is not None:
            self.counter.update(incr=incr)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and enlighten is not None:
            if self.counter is not None:
                self.counter.__exit__(exc_type, exc_val, exc_tb)
            if self.manager is not None:
                return self.manager.__exit__(exc_type, exc_val, exc_tb)
