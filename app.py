import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from visualize import demo  # noqa: F401
