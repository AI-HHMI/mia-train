"""Run a command with its CPUs and memory on ONE NUMA node: the node holding most of this job's CPUs.

  numa_local.py CMD [ARGS...]

Why (measured 2026-09-27 on the 6a mws_fill scoring job, h06u06): LSF gave the 32-slot job cores on
both sockets, the single-threaded mutex watershed ran on node 1 with 22% of its 199 GB on node 0, and
the kernel's automatic NUMA balancing (numa_balancing=1) kept unmapping its pages to sample which
socket touched them. The watershed's random access turned that into 418k minor faults a second and
73% of the process's CPU time in the kernel: the same watershed took more than twice as long as on
another node. A process with an explicit memory policy is not scanned, so binding the CPUs to one
node and preferring its memory removes both the remote accesses and the faults. `--preferred`, not
`--membind`: if the node fills, allocations spill to the other node instead of failing.
"""

from __future__ import annotations

import glob
import os
import sys


def cpulist(path: str) -> set[int]:
    cpus: set[int] = set()
    for part in open(path).read().strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


allowed = os.sched_getaffinity(0)
nodes = {
    int(path.rsplit("node", 1)[1]): cpulist(f"{path}/cpulist") & allowed
    for path in glob.glob("/sys/devices/system/node/node[0-9]*")
}
node = max(nodes, key=lambda n: len(nodes[n]))
print(f"[numa_local] NUMA node {node}: {len(nodes[node])} of the job's {len(allowed)} CPUs", flush=True)
os.execvp("numactl", ["numactl", f"--cpunodebind={node}", f"--preferred={node}", *sys.argv[1:]])
