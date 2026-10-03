"""Cycle-level simulation of rmsnorm_qk variants with FlashGPU-Sim (CPU only, no GPU).

Per variant and workload:
  1. capture: run the variant's run() on CPU tensors under TritonTrace offline mode, which
     compiles each Triton launch for the chosen architecture and writes a standalone harness;
  2. simulate: build each harness and replay it under FlashGPU-Sim with a GPU config;
  3. record gpu_tot_sim_cycle (summed over launches) and convert to seconds with the config clock.

Usage (one variant per call so a Slurm array can fan out):
  python simulate.py --vid v000 --config SM90_H100 --arch sm90
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLX = Path(os.environ.get("SOLX", HERE.parent))
SIM = SOLX / "flashgpusim"
WORKLOADS = [(1, 128), (1, 131), (2, 128), (1, 256)]  # the four smallest; simulation time grows with size

CAPTURE = r"""
import json, sys, torch, TritonTrace
sys.path.insert(0, {here!r})
import static_features
mod = static_features.load_variant({sol!r})
mod._NUM_SMS = {sms}
mod.EVICT = False  # the simulator's PTX parser rejects createpolicy (cache hints), and would not model them anyway
b, s = {b}, {s}
q = torch.randn(b, s, 48, 128); k = torch.randn(b, s, 48, 128)
wq = torch.randn(48, 128); wk = torch.randn(48, 128)
qo = torch.empty_like(q); ko = torch.empty_like(k)
tracker = TritonTrace.Tracker({out!r}, mode="offline", target={arch!r})
mod.run(q, k, wq, wk, 1e-6, qo, ko)
tracker.save_summary()
"""


def config_value(config, key):
    text = (SIM / "configs" / config / "gpgpusim.config").read_text()
    return re.findall(rf"^-{key}\s+(\S+)", text, re.M)[-1]


def config_clock_hz(config):
    return float(config_value(config, "gpgpu_clock_domains").split(":")[0]) * 1e6


def config_sms(config):
    return int(config_value(config, "gpgpu_n_clusters")) * int(config_value(config, "gpgpu_n_cores_per_cluster"))


def capture(vid, b, s, arch, sms, out):
    if out.exists():
        shutil.rmtree(out)
    code = CAPTURE.format(here=str(HERE), sol=str(HERE / "variants" / f"{vid}.json"), sms=sms, b=b, s=s,
                          out=str(out), arch=arch)
    env = {k: v for k, v in os.environ.items() if "flashgpusim" not in v}  # capture must not see the simulator libs
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if p.returncode:
        raise RuntimeError("capture failed:\n" + p.stderr[-3000:])
    return sorted((out / "launchers").glob("*_launch*_Makefile"))


def simulate(makefile, config, cuda_home):
    launchers = makefile.parent
    shutil.copytree(SIM / "configs" / config, launchers, dirs_exist_ok=True)
    name = makefile.name.removesuffix("_Makefile")
    script = (
        f"export CUDA_INSTALL_PATH={cuda_home}; export PATH={cuda_home}/bin:$PATH; "
        f"cd {SIM} && source setup_environment >/dev/null && cd {launchers} && "
        f"make --no-print-directory -f {makefile.name} >/dev/null && "
        f"export OMP_NUM_THREADS={os.environ.get('SLURM_CPUS_PER_TASK', '4')} && ./{name}"
    )
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    (launchers / f"{name}.simlog").write_text(p.stdout + "\n---\n" + p.stderr)
    m = re.search(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", p.stdout, re.M)
    if not m:
        raise RuntimeError(f"no cycle count for {name}:\n" + (p.stdout + p.stderr)[-3000:])
    stats = {"cycles": int(m[1])}
    for key in ("gpu_tot_ipc", "L2_BW_total", "gpgpu_simulation_time"):
        mm = re.search(rf"^{key}\s*=\s*(.+)$", p.stdout, re.M)
        if mm:
            stats[key] = mm[1].strip()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", required=True)
    ap.add_argument("--config", default="SM90_H100")
    ap.add_argument("--arch", default="sm90")
    ap.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME", "/opt/apps/libs/nvidia-cuda/toolkit/12.8.1"))
    ap.add_argument("--out", type=Path, default=HERE / "results" / "sim")
    args = ap.parse_args()

    clock = config_clock_hz(args.config)
    result_file = args.out / args.config / f"{args.vid}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    if result_file.exists() and all("sim_ms" in r for r in json.loads(result_file.read_text())):
        print(f"{args.vid}: already simulated")
        return
    results = []
    for b, s in WORKLOADS:
        work = SOLX / "simwork" / args.config / f"{args.vid}_b{b}_s{s}"
        rec = dict(vid=args.vid, config=args.config, batch_size=b, seq_len=s)
        try:
            makefiles = capture(args.vid, b, s, args.arch, config_sms(args.config), work / "tracking")
            per_launch = [simulate(mf, args.config, args.cuda_home) for mf in makefiles]
            rec.update(launches=len(per_launch), cycles=sum(x["cycles"] for x in per_launch),
                       sim_ms=sum(x["cycles"] for x in per_launch) / clock * 1e3, detail=per_launch)
        except Exception as e:
            rec["error"] = str(e)[-2000:]
        print(json.dumps({k: v for k, v in rec.items() if k != "detail"}), flush=True)
        results.append(rec)
    result_file.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
