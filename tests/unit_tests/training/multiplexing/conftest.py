"""Load the dependency-free transport package without importing all model families."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "src/megatron/bridge/training/multiplexing"
spec = importlib.util.spec_from_file_location(
    "multiplexing_subject", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
