"""Keep a training run to a share of a shared machine.

Configured by the `resource_limits` block of a config (every key optional, missing / null = no limit):

    resource_limits:
      gpu_mem_fraction: 0.48 # cap on the PyTorch allocator, as a fraction of total VRAM (+ ~0.5 GB CUDA context)
      ram_gb: 16             # hard cap on committed memory of the whole job; allocations beyond it fail
      cpu_percent: null      # hard cap on total CPU time of this process + its DataLoader workers
      gpu_duty: null         # fraction of wall time the GPU may be busy; the loop sleeps for the rest

On Windows the CPU / RAM caps use a Job Object (no admin rights needed), which also runs the job at
below-normal priority and kills the DataLoader workers if the main process dies. Elsewhere the CPU cap
falls back to CPU affinity + nice.
"""
import os
import sys
import time

import torch

_job = None  # Windows job handle; kept for the lifetime of the process

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    class _CpuRateInfo(ctypes.Structure):
        _fields_ = [('ControlFlags', wintypes.DWORD), ('CpuRate', wintypes.DWORD)]

    class _BasicLimitInfo(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class _ExtendedLimitInfo(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', _BasicLimitInfo), ('IoInfo', ctypes.c_ulonglong * 6),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _kernel32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                                    wintypes.DWORD, wintypes.LPDWORD]
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]

    _EXTENDED_LIMIT_INFO, _CPU_RATE_INFO = 9, 15
    _LIMIT_PRIORITY_CLASS, _LIMIT_JOB_MEMORY, _LIMIT_KILL_ON_JOB_CLOSE = 0x20, 0x200, 0x2000
    _BELOW_NORMAL_PRIORITY_CLASS = 0x4000
    _CPU_RATE_ENABLE, _CPU_RATE_HARD_CAP = 0x1, 0x4

    def _check(ok, what):
        if not ok:
            raise OSError(ctypes.get_last_error(), '{} failed'.format(what))

    def _windows_job(cpu_percent, ram_gb):
        global _job
        job = _kernel32.CreateJobObjectW(None, None)
        _check(job, 'CreateJobObject')
        info = _ExtendedLimitInfo()
        info.BasicLimitInformation.LimitFlags = _LIMIT_PRIORITY_CLASS | _LIMIT_KILL_ON_JOB_CLOSE
        info.BasicLimitInformation.PriorityClass = _BELOW_NORMAL_PRIORITY_CLASS
        if ram_gb:
            info.BasicLimitInformation.LimitFlags |= _LIMIT_JOB_MEMORY
            info.JobMemoryLimit = int(ram_gb * 2 ** 30)
        _check(_kernel32.SetInformationJobObject(job, _EXTENDED_LIMIT_INFO, ctypes.byref(info),
                                                 ctypes.sizeof(info)), 'SetInformationJobObject(limits)')
        if cpu_percent:
            rate = _CpuRateInfo(_CPU_RATE_ENABLE | _CPU_RATE_HARD_CAP, int(cpu_percent * 100))
            _check(_kernel32.SetInformationJobObject(job, _CPU_RATE_INFO, ctypes.byref(rate),
                                                     ctypes.sizeof(rate)), 'SetInformationJobObject(cpu rate)')
        _check(_kernel32.AssignProcessToJobObject(job, _kernel32.GetCurrentProcess()), 'AssignProcessToJobObject')
        _job = job


def apply(limits, device):
    """Apply the limits to this process (call before the DataLoader workers start). Returns log lines."""
    limits = limits or {}
    cpu, ram = limits.get('cpu_percent'), limits.get('ram_gb')
    lines = []
    if cpu:
        torch.set_num_threads(max(1, int(os.cpu_count() * cpu / 100)))
    if sys.platform == 'win32':
        try:
            _windows_job(cpu, ram)
            lines.append('CPU {}, RAM {}, below-normal priority (Windows job object)'.format(
                'hard cap {}% of all cores'.format(cpu) if cpu else 'not capped',
                'hard cap {} GB'.format(ram) if ram else 'not capped'))
        except OSError as e:
            lines.append('WARNING: could not create the job object ({}); CPU / RAM are NOT capped'.format(e))
    else:
        if cpu:
            cores = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, cores[:max(1, int(len(cores) * cpu / 100))])
            lines.append('CPU limited to {} of {} cores (affinity)'.format(len(os.sched_getaffinity(0)), len(cores)))
        os.nice(10)
        if ram:
            lines.append('WARNING: ram_gb is only enforced on Windows')
    frac = limits.get('gpu_mem_fraction')
    if frac and device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(frac, device)
        lines.append('GPU memory cap {:.0%} of {:.1f} GB'.format(
            frac, torch.cuda.get_device_properties(device).total_memory / 1e9))
    duty = limits.get('gpu_duty')
    if duty and duty < 1 and device.type == 'cuda':
        lines.append('GPU busy at most {:.0%} of the time'.format(duty))
    return lines


def peak_usage(device):
    """Peak memory so far in GB: {'ram_gb': committed RAM of the job (Windows only), 'gpu_gb': reserved VRAM}."""
    peaks = {}
    if _job is not None:
        info = _ExtendedLimitInfo()
        if _kernel32.QueryInformationJobObject(_job, _EXTENDED_LIMIT_INFO, ctypes.byref(info),
                                               ctypes.sizeof(info), None):
            peaks['ram_gb'] = round(info.PeakJobMemoryUsed / 2 ** 30, 2)
    if device.type == 'cuda':
        peaks['gpu_gb'] = round(torch.cuda.max_memory_reserved(device) / 1e9, 2)
    return peaks


def usage(device):
    """Peak memory so far, for the log."""
    peaks = peak_usage(device)
    parts = []
    if 'ram_gb' in peaks:
        parts.append('peak committed RAM {:.1f} GB'.format(peaks['ram_gb']))
    if 'gpu_gb' in peaks:
        parts.append('peak GPU memory {:.1f} GB reserved'.format(peaks['gpu_gb']))
    return ', '.join(parts)


def keep_awake():
    """Stop Windows from going to sleep on idle while this process runs."""
    if sys.platform == 'win32':
        _kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


class GpuThrottle:
    """Sleeps after each step so the GPU is busy for at most `duty` of the wall time."""

    def __init__(self, duty, device):
        self.duty = duty
        self.enabled = bool(duty) and duty < 1 and device.type == 'cuda'

    def start(self):
        if self.enabled:
            torch.cuda.synchronize()
            self.t = time.perf_counter()

    def stop(self):
        if self.enabled:
            torch.cuda.synchronize()
            time.sleep((time.perf_counter() - self.t) * (1 - self.duty) / self.duty)
