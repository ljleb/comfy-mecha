import pathlib
import importlib.util
import sys
import traceback

extensions_path = pathlib.Path(__file__).parent / "mecha_extensions"
for path in extensions_path.iterdir():
    if path.is_file() and path.suffix == ".py":
        try:
            module_spec = importlib.util.spec_from_file_location(path.stem, str(path))
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        except Exception as e:
            print(f"[comfy-mecha] could not import mecha extension {path}: {e}", file=sys.stderr)
            traceback.print_exc()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "js"
