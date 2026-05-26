"""Re-export UNetModel via BBDM's openaimodel (no BBDM files modified)."""
import sys
from pathlib import Path

_bbdm_root = str(Path(__file__).resolve().parents[2] / "BBDM")
if _bbdm_root not in sys.path:
    sys.path.insert(0, _bbdm_root)

from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel  # noqa: E402

__all__ = ["UNetModel"]
