"""Make src/ importable with flat imports (as the entry scripts do)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
