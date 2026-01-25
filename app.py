import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from visualize import build_demo 

demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
