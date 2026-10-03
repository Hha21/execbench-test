"""Crude analytical time model for the rmsnorm_qk variants on a given GPU.

The kernel is memory-bound, so the estimate is bytes / achievable bandwidth, where
achievable bandwidth is capped either by the peak or by how many bytes the resident
programs can keep in flight (Little's law). Occupancy uses register and shared-memory
counts from the static features for that GPU's architecture. Constants are public specs
or rough guesses; the emulator learns how wrong this model is.
"""

import math

GPUS = {
    #        SMs  bytes/s   threads/SM blocks/SM regs/SM  smem/SM   DRAM latency (s)  arch
    "A100": dict(sms=108, bw=2.039e12, threads=2048, blocks=32, regs=65536, smem=167936, lat=600e-9, arch="sm_80"),
    "L40S": dict(sms=142, bw=0.864e12, threads=1536, blocks=24, regs=65536, smem=102400, lat=600e-9, arch="sm_89"),
    "H200": dict(sms=132, bw=4.8e12, threads=2048, blocks=32, regs=65536, smem=233472, lat=650e-9, arch="sm_90"),
    "B200": dict(sms=148, bw=8.0e12, threads=2048, blocks=32, regs=65536, smem=233472, lat=700e-9, arch="sm_100a"),
}

D = 128
H = 48
LAUNCH_S = 2e-6  # per-launch fixed cost (ramp-up and, for two launches, the gap between them)


def blocks_per_sm(gpu, num_warps, regs, smem):
    g = GPUS[gpu]
    threads = 32 * num_warps
    limits = [g["blocks"], g["threads"] // threads]
    if regs:
        limits.append(g["regs"] // (regs * threads))
    if smem:
        limits.append(g["smem"] // smem)
    return max(1, min(limits))


def estimate(gpu, knobs, axes, regs, smem):
    """Estimated seconds for one call, plus intermediate quantities useful as features."""
    g = GPUS[gpu]
    n_rows = axes["batch_size"] * axes["seq_len"] * H
    launches = 1 if knobs["FUSE"] else 2
    tensors_per_launch = 2 if knobs["FUSE"] else 1
    bytes_per_launch = tensors_per_launch * n_rows * D * 4 * 2  # read + write, fp32
    tiles = math.ceil(n_rows / knobs["ROWS"]) * tensors_per_launch
    programs = min(tiles, g["sms"] * knobs["PROGS_PER_SM"]) if knobs["PERSIST"] else tiles
    bps = blocks_per_sm(gpu, knobs["NUM_WARPS"], regs, smem)
    capacity = g["sms"] * bps
    resident = min(programs, capacity)
    in_flight = resident * knobs["ROWS"] * D * 4
    bw = min(g["bw"], in_flight / g["lat"])
    waves = programs / capacity
    quantisation = math.ceil(waves) / waves if waves > 1 and not knobs["PERSIST"] else 1.0
    t = launches * (bytes_per_launch / bw * quantisation + LAUNCH_S)
    return {
        "t_analytic": t,
        "bytes": launches * bytes_per_launch,
        "occupancy_blocks": bps,
        "resident_frac": resident / capacity,
        "bw_frac": bw / g["bw"],
        "waves": waves,
    }
