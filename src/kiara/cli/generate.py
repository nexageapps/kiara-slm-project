"""CLI for text generation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.generate import main

if __name__ == "__main__":
    main()
