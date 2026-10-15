"""Filesystem roots shared by editable and wheel installations."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
