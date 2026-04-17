from __future__ import annotations

import os
import threading
import time

import torch


class WandbSystemMonitor:
    """Periodic system resource logger for a W&B run."""

    def __init__(
        self,
        wandb_run: object,
        interval_sec: float = 2.0,
        prefix: str = "system",
    ) -> None:
        self.wandb_run = wandb_run
        self.interval_sec = float(max(0.5, interval_sec))
        self.prefix = str(prefix)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._psutil = None
        self._proc = None
        self._cpu_count = 1

        self._pynvml = None
        self._nvml_handles: list[object] = []
        self._nvml_ready = False

        self._init_backends()

    @staticmethod
    def _to_gib(x: int | float) -> float:
        return float(x) / (1024.0 ** 3)

    def _init_backends(self) -> None:
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._cpu_count = int(max(1, psutil.cpu_count(logical=True) or 1))
            self._proc = psutil.Process(os.getpid())
            # Warm-up for non-zero cpu_percent deltas.
            psutil.cpu_percent(interval=None)
            self._proc.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
            self._proc = None
            self._cpu_count = 1

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            n_gpu = int(pynvml.nvmlDeviceGetCount())
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpu)]
            self._pynvml = pynvml
            self._nvml_ready = n_gpu > 0
        except Exception:
            self._pynvml = None
            self._nvml_handles = []
            self._nvml_ready = False

    def snapshot(self) -> dict[str, float]:
        p = self.prefix
        out: dict[str, float] = {f"{p}/timestamp": float(time.time())}

        if self._psutil is not None:
            psutil = self._psutil
            try:
                out[f"{p}/cpu/percent"] = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            try:
                vm = psutil.virtual_memory()
                out[f"{p}/ram/percent"] = float(vm.percent)
                out[f"{p}/ram/used_gb"] = self._to_gib(vm.used)
                out[f"{p}/ram/available_gb"] = self._to_gib(vm.available)
                out[f"{p}/ram/total_gb"] = self._to_gib(vm.total)
            except Exception:
                pass
            if self._proc is not None:
                try:
                    proc_cpu_total = float(self._proc.cpu_percent(interval=None))
                    out[f"{p}/process/cpu_percent_total"] = proc_cpu_total
                    out[f"{p}/process/cpu_percent_norm"] = proc_cpu_total / float(max(1, self._cpu_count))
                except Exception:
                    pass
                try:
                    rss = int(self._proc.memory_info().rss)
                    out[f"{p}/process/ram_gb"] = self._to_gib(rss)
                except Exception:
                    pass

        # CUDA allocator stats from torch.
        if torch.cuda.is_available():
            try:
                n_cuda = int(torch.cuda.device_count())
            except Exception:
                n_cuda = 0
            alloc_sum = 0.0
            reserve_sum = 0.0
            for i in range(n_cuda):
                try:
                    alloc = self._to_gib(torch.cuda.memory_allocated(i))
                    reserved = self._to_gib(torch.cuda.memory_reserved(i))
                except Exception:
                    continue
                out[f"{p}/cuda/{i}/mem_alloc_gb"] = alloc
                out[f"{p}/cuda/{i}/mem_reserved_gb"] = reserved
                alloc_sum += alloc
                reserve_sum += reserved
            if n_cuda > 0:
                out[f"{p}/cuda/mem_alloc_total_gb"] = alloc_sum
                out[f"{p}/cuda/mem_reserved_total_gb"] = reserve_sum

        # NVML stats (GPU util/memory/power/temp).
        if self._nvml_ready and self._pynvml is not None:
            pynvml = self._pynvml
            util_vals: list[float] = []
            power_sum = 0.0
            mem_used_sum = 0.0
            mem_total_sum = 0.0
            for i, handle in enumerate(self._nvml_handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    out[f"{p}/gpu/{i}/util_percent"] = float(util.gpu)
                    out[f"{p}/gpu/{i}/mem_util_percent"] = float(util.memory)
                    out[f"{p}/gpu/{i}/mem_used_gb"] = self._to_gib(mem.used)
                    out[f"{p}/gpu/{i}/mem_total_gb"] = self._to_gib(mem.total)
                    out[f"{p}/gpu/{i}/power_w"] = power_w
                    try:
                        out[f"{p}/gpu/{i}/temp_c"] = float(
                            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        )
                    except Exception:
                        pass
                    try:
                        out[f"{p}/gpu/{i}/power_limit_w"] = float(
                            pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
                        ) / 1000.0
                    except Exception:
                        pass
                    util_vals.append(float(util.gpu))
                    power_sum += power_w
                    mem_used_sum += self._to_gib(mem.used)
                    mem_total_sum += self._to_gib(mem.total)
                except Exception:
                    continue
            if util_vals:
                out[f"{p}/gpu/count"] = float(len(util_vals))
                out[f"{p}/gpu/util_mean_percent"] = float(sum(util_vals) / max(1, len(util_vals)))
                out[f"{p}/gpu/power_total_w"] = float(power_sum)
                out[f"{p}/gpu/mem_used_total_gb"] = float(mem_used_sum)
                out[f"{p}/gpu/mem_total_gb"] = float(mem_total_sum)
                if mem_total_sum > 1e-9:
                    out[f"{p}/gpu/mem_total_percent"] = float(100.0 * mem_used_sum / mem_total_sum)

        return out

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            try:
                self.wandb_run.log(self.snapshot())
            except Exception:
                break

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="wandb-system-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(2.0, 2.0 * self.interval_sec + 0.5))
            self._thread = None
        try:
            self.wandb_run.log(self.snapshot())
        except Exception:
            pass
        if self._nvml_ready and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
