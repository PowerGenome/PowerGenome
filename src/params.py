"""
Parameters and settings
"""

from pathlib import Path
from src import __file__

# Not convinced this is the best way to set folder paths but it works!
src_path = Path(__file__).parent
project_path = src_path.parent

DATA_PATHS = {}
DATA_PATHS["results"] = project_path / "results"
DATA_PATHS['src'] = project_path / "src"
