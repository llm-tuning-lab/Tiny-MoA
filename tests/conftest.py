"""Pytest configuration for Tiny-MoA tests."""

import sys
from pathlib import Path

# Add project root to sys.path so tests can import project modules (e.g., src.tools)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
