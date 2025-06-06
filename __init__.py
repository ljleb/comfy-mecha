import pathlib
import importlib.util
import sys
import traceback
from types import ModuleType

extensions_path = pathlib.Path(__file__).parent / "mecha_extensions"
sys.modules["mecha_extensions"] = ModuleType("mecha_extensions")
for path in extensions_path.iterdir():
    is_script = path.is_file() and path.suffix == ".py"
    is_package = path.is_dir() and (path / "__init__.py").is_file()

    if not (is_script or is_package):
        continue

    if is_package:
        script_path = path / "__init__.py"
        module_name = f"mecha_extensions.{path.name}"
    else:
        script_path = path
        module_name = f"mecha_extensions.{path.stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create a module spec for {script_path!r}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[comfy-mecha] could not import mecha extension {path}: {e}", file=sys.stderr)
        traceback.print_exc()


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "js"
