# iris_td/training/run_ddpm_scoring.py
"""
Run DDPM inference for all ablation combinations.

Ablation Study 1 — diffusion type comparison:
  Row 1: gaussian, t*=1000, steps=100, no DDIM  — vanilla DDPM
  Row 2: gaussian, t*=500,  steps=100, no DDIM  — partial noise only
  Row 3: simplex,  t*=500,  steps=100, no DDIM  — AnoDDPM
  Row 4: simplex,  t*=500,  steps=100, DDIM     — AnoDDPM + DDIM

Ablation Study 2 — efficiency vs accuracy:
  Best config (row 3 or 4) at steps: 10, 25, 50, 100

Run val split first — tune — then test split.

DO NOT RUN — GPU is busy.
Add --run flag when GPU is free.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ablation 1 — fixed step count, vary noise + t* + ddim
ABLATION1_CONFIGS = [
    # (noise_type, t_star, use_ddim, label)
    ("gaussian", 1000, False, "vanilla_ddpm"),
    ("gaussian",  500, False, "partial_gaussian"),
    ("simplex",   500, False, "anoddpm_simplex"),
    ("simplex",   500, True,  "anoddpm_simplex_ddim"),
]

# Ablation 2 — vary steps (use best config from ablation 1)
STEP_COUNTS = [10, 25, 50, 100]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        default="iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth")
    parser.add_argument("--config",
        default="iris_td/configs/ddpm_iris.yaml")
    parser.add_argument("--split",
        choices=["test", "val"], default="val")
    parser.add_argument("--lambda_lpips", type=float, default=0.5)
    parser.add_argument("--best_config",
        default="anoddpm_simplex",
        help="Label of best ablation1 config for ablation2 step sweep")
    parser.add_argument("--run",
        action="store_true",
        help="Actually execute commands. Without flag, just prints them.")
    return parser.parse_args()


def build_cmd(checkpoint, config, test_dir, output_csv,
              noise_type, t_star, num_steps, lambda_lpips,
              split, use_ddim):
    cmd = [
        sys.executable,
        "iris_td/models/ddpm_anomaly_detector.py",
        "--config",       config,
        "--checkpoint",   checkpoint,
        "--test_dir",     test_dir,
        "--output_csv",   output_csv,
        "--labels",       f"iris_td/labels/{split}_labels.csv",
        "--noise_type",   noise_type,
        "--t_star",       str(t_star),
        "--num_steps",    str(num_steps),
        "--lambda_lpips", str(lambda_lpips),
        "--split",        split,
    ]
    if use_ddim:
        cmd.append("--use_ddim")
    return cmd


def main():
    args     = parse_args()
    test_dir = f"iris_td/data/evaluation_sets/{args.split}/"
    env      = os.environ.copy()
    env["PYTHONPATH"] = (str(Path("BBDM").resolve()) +
                         os.pathsep + env.get("PYTHONPATH", ""))
    commands = []

    # ── Ablation 1 ────────────────────────────────────────────────────
    log.info("Ablation 1 commands:")
    for noise, t_star, use_ddim, label in ABLATION1_CONFIGS:
        csv_out = (f"iris_td/pad_scores/"
                   f"ddpm_{args.split}_{label}_steps100.csv")
        cmd = build_cmd(
            args.checkpoint, args.config, test_dir, csv_out,
            noise, t_star, 100, args.lambda_lpips,
            args.split, use_ddim
        )
        commands.append((f"abl1_{label}", cmd))
        log.info(f"  {label} → {csv_out}")

    # ── Ablation 2 ────────────────────────────────────────────────────
    # Find best config params
    best = next((c for c in ABLATION1_CONFIGS if c[3] == args.best_config),
                ABLATION1_CONFIGS[2])
    best_noise, best_t, best_ddim, _ = best

    log.info(f"Ablation 2 commands (best={args.best_config}):")
    for steps in STEP_COUNTS:
        csv_out = (f"iris_td/pad_scores/"
                   f"ddpm_{args.split}_{args.best_config}_steps{steps}.csv")
        cmd = build_cmd(
            args.checkpoint, args.config, test_dir, csv_out,
            best_noise, best_t, steps, args.lambda_lpips,
            args.split, best_ddim
        )
        commands.append((f"abl2_steps{steps}", cmd))
        log.info(f"  steps={steps} → {csv_out}")

    # Print all commands
    print("\n" + "=" * 60)
    print("COMMANDS — run when GPU is free:")
    print("=" * 60)
    for name, cmd in commands:
        print(f"\n# {name}")
        print("PYTHONPATH=BBDM:$PYTHONPATH " + " ".join(cmd))

    if args.run:
        for name, cmd in commands:
            log.info(f"Running {name}...")
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                log.error(f"FAILED: {name}")
                break
    else:
        print("\nAdd --run to execute. Run val first, then test.")


if __name__ == "__main__":
    main()
