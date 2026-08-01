#!/usr/bin/env python3
"""Run a command and report wall time + peak RSS to a stats file.

Stands in for `/usr/bin/time -v`, which isn't installed on every box. Peak RSS
comes from getrusage(RUSAGE_CHILDREN) after the child exits, and is sampled
from /proc while it runs so we also see the high-water mark of the whole tree.

  usage: runstat.py <stats-file> <cmd> [args ...]
"""
import os
import resource
import subprocess
import sys
import threading
import time


def sample_peak(pid, out, stop):
    """Poll VmHWM so we catch the tree's high-water mark, not just the reaped child."""
    while not stop.is_set():
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        out[0] = max(out[0], int(line.split()[1]))
                        break
        except (OSError, ValueError):
            return
        stop.wait(0.5)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    stats_path, cmd = sys.argv[1], sys.argv[2:]

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    peak = [0]
    stop = threading.Event()
    sampler = threading.Thread(target=sample_peak, args=(proc.pid, peak, stop), daemon=True)
    sampler.start()
    rc = proc.wait()
    stop.set()
    sampler.join(timeout=2)
    wall = time.monotonic() - t0

    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_kb = max(peak[0], ru.ru_maxrss)  # ru_maxrss is KiB on Linux
    with open(stats_path, "w") as fh:
        fh.write(f"exit_code {rc}\n")
        fh.write(f"wall_s {wall:.2f}\n")
        fh.write(f"peak_rss_kb {peak_kb}\n")
        fh.write(f"peak_rss_gb {peak_kb / 1048576:.2f}\n")
        fh.write(f"user_s {ru.ru_utime:.2f}\n")
        fh.write(f"sys_s {ru.ru_stime:.2f}\n")
        fh.write(f"major_faults {ru.ru_majflt}\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
