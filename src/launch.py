from kernel.command_module import command_module
from kernel.config import config
from kernel.kernel import kernel

from modules.module_core import module_core
from modules.module_io import module_io

import argparse
import json

MODULES_REGISTRY: dict[str, command_module] = {
    "core": module_core(),
    "io": module_io(),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Launch the Pulsar Project kernel.")
    parser.add_argument(
        "--modules",
        nargs="*",
        help="List of module names to load (e.g., --modules core io)",
    )
    parser.add_argument(
        "--modules-json",
        type=str,
        help="Path to a JSON file containing a list of module names.",
    )
    parser.add_argument(
        "--initial-task",
        type=str,
        help="Specify the initial task to execute after kernel initialization.",
    )
    args = parser.parse_args()

    # Load modules
    if args.modules_json:
        with open(args.modules_json, "r") as f:
            module_names = json.load(f)  # TODO: Make a JSON file
            if not isinstance(module_names, list) or not all(
                isinstance(name, str) for name in module_names
            ):
                raise ValueError("JSON file must contain a list of module names.")
    else:
        module_names = args.modules or []
    for module in module_names:
        if module not in MODULES_REGISTRY:
            raise ValueError(f"Module '{module}' not found in registry.")
        else:
            kernel.add_module(MODULES_REGISTRY[module])

    # Specify initial task
    config.INITIAL_TASK = args.initial_task

    return module_names


###! Main
if __name__ == "__main__":
    module_names = parse_arguments()
    kernel.thinking_loop()
