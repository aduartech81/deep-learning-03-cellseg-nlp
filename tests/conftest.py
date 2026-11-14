import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ.setdefault("DL03_PROFILE", "smoke")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
