#!/usr/bin/env python3
"""Smoke-test the iris_td DDPM pipeline end-to-end without real data.

Checks:
  1. YAML config loads cleanly (no !!python/tuple, all required keys present)
  2. LatentDDPM instantiates (VQGAN + UNet + schedule buffers)
  3. encode():          (1,3,256,256) → (1,3,64,64)
  4. decode():          (1,3,64,64)   → (1,3,256,256)
  5. q_sample():        returns (z_t, noise) with correct shapes
  6. predict_noise():   (1,3,64,64) → (1,3,64,64)
  7. p_sample():        (1,3,64,64) → (1,3,64,64)
  8. ddim_sample():     (1,3,64,64) → (1,3,64,64)
  9. alphas_cumprod buffer exists and is monotonically decreasing

Usage:
    python iris_td/scripts/verify_pipeline.py
"""

import sys
import traceback
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IRIS_TD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IRIS_TD_ROOT / "models"))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SEP  = "-" * 60


def check(name: str, fn):
    """Run fn(); print PASS or FAIL with traceback on failure."""
    try:
        result = fn()
        print(f"{PASS} {name}")
        return result
    except Exception:
        print(f"{FAIL} {name}")
        traceback.print_exc()
        return None


def main():
    print(SEP)
    print("iris_td DDPM pipeline verification")
    print(SEP)

    fails = []

    # ------------------------------------------------------------------
    # 1. Load YAML
    # ------------------------------------------------------------------
    cfg_path = IRIS_TD_ROOT / "configs" / "ddpm_iris.yaml"

    def load_yaml():
        assert cfg_path.exists(), f"Config not found: {cfg_path}"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for key in ("unet_config", "vqgan_config", "ddpm_config"):
            assert key in cfg["model"], f"Missing cfg['model']['{key}']"
        return cfg

    cfg = check("YAML loads — required keys present", load_yaml)
    if cfg is None:
        fails.append("YAML load")

    if cfg is None:
        print("\n[ABORT] Cannot continue without config.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Instantiate LatentDDPM (CPU — no GPU needed for shape checks)
    # ------------------------------------------------------------------
    device = torch.device("cpu")

    def make_model():
        from ddpm_model import LatentDDPM  # noqa: PLC0415
        m = LatentDDPM(
            unet_config=cfg["model"]["unet_config"],
            vqgan_config=cfg["model"]["vqgan_config"],
            ddpm_config=cfg["model"]["ddpm_config"],
        )
        m.to(device)
        m.eval()
        return m

    model = check("LatentDDPM instantiates", make_model)
    if model is None:
        fails.append("LatentDDPM init")

    if model is None:
        print("\n[ABORT] Model failed to instantiate.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3–8. Shape checks (tiny tensors on CPU, fast)
    # ------------------------------------------------------------------
    pixel = torch.randn(1, 3, 256, 256, device=device)
    latent = torch.randn(1, 3, 64, 64, device=device)
    t_batch = torch.tensor([100], dtype=torch.long, device=device)

    # 3. encode
    def check_encode():
        z = model.encode(pixel)
        assert z.shape == (1, 3, 64, 64), f"encode output shape {z.shape}"
        return z

    z = check("encode: (1,3,256,256) → (1,3,64,64)", check_encode)
    if z is None:
        fails.append("encode shape")
        z = latent  # fallback so downstream checks can continue

    # 4. decode
    def check_decode():
        x = model.decode(latent)
        assert x.shape == (1, 3, 256, 256), f"decode output shape {x.shape}"
        return True

    if check("decode: (1,3,64,64) → (1,3,256,256)", check_decode) is None:
        fails.append("decode shape")

    # 5. q_sample
    def check_q_sample():
        z_t, noise = model.q_sample(latent, t_batch)
        assert z_t.shape == (1, 3, 64, 64), f"z_t shape {z_t.shape}"
        assert noise.shape == (1, 3, 64, 64), f"noise shape {noise.shape}"
        return z_t

    z_t = check("q_sample: latent + timestep → z_t, noise", check_q_sample)
    if z_t is None:
        fails.append("q_sample")
        z_t = latent

    # 6. predict_noise
    def check_predict_noise():
        eps = model.predict_noise(z_t, t_batch)
        assert eps.shape == (1, 3, 64, 64), f"predict_noise shape {eps.shape}"
        return True

    if check("predict_noise: (z_t, t) → (1,3,64,64)", check_predict_noise) is None:
        fails.append("predict_noise")

    # 7. p_sample
    def check_p_sample():
        z_prev = model.p_sample(z_t, t_batch)
        assert z_prev.shape == (1, 3, 64, 64), f"p_sample shape {z_prev.shape}"
        return True

    if check("p_sample: z_t → z_{t-1}", check_p_sample) is None:
        fails.append("p_sample")

    # 8. ddim_sample
    def check_ddim_sample():
        z_prev = model.ddim_sample(z_t, t_batch, eta=0.0)
        assert z_prev.shape == (1, 3, 64, 64), f"ddim_sample shape {z_prev.shape}"
        return True

    if check("ddim_sample: z_t → z_{t-1} (deterministic)", check_ddim_sample) is None:
        fails.append("ddim_sample")

    # 9. alphas_cumprod
    def check_schedule():
        ac = model.alphas_cumprod
        assert ac.shape[0] == cfg["model"]["ddpm_config"]["num_timesteps"]
        assert (ac[:-1] >= ac[1:]).all(), "alphas_cumprod not monotonically decreasing"
        assert ac[0] > 0.99, f"alphas_cumprod[0]={ac[0]:.4f} (should be ≈1)"
        assert ac[-1] < 0.01, f"alphas_cumprod[-1]={ac[-1]:.4f} (should be ≈0)"
        return True

    if check("alphas_cumprod buffer: shape, monotone, range", check_schedule) is None:
        fails.append("alphas_cumprod")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(SEP)
    if not fails:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILED CHECKS ({len(fails)}): {', '.join(fails)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
