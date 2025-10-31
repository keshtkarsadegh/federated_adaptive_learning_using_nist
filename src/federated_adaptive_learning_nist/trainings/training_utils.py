import importlib.util
import inspect
import pathlib
import sys

from src.federated_adaptive_learning_nist.nist_logger import NistLogger


def find_trainer_by_name(class_name):
    parent_path = pathlib.Path("src/federated_adaptive_learning_nist/trainers")



    # Scan all Python files in the folder
    for py_file in parent_path.glob("*.py"):
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for the class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name == class_name and obj.__module__ == module.__name__:
                    return obj
    return None

def get_trainer_class(class_name: str):
    import importlib, inspect, pkgutil

    pkg_name = "src.federated_adaptive_learning_nist.trainers"
    pkg = importlib.import_module(pkg_name)

    for _, modname, is_pkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        if is_pkg:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            NistLogger.info("Found trainer class: {}".format(class_name))
            return cls

    raise ImportError(f"Class '{class_name}' not found under '{pkg_name}'.")
