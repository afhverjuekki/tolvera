"""Sketchbook module for listing and running sketches from the command line.

WIP.
"""

import datetime
import importlib.util
import os
import random
import sys
from typing import Any, Dict, List

import fire


def list_sketches(
    sketchbook_folder: str = "./", sort: str = "name", direction: str = "ascending"
) -> None:
    """
    Lists all sketches in the given sketchbook folder.

    Args:
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.
        sort (str): Sort sketches by name, size, modified or created. Defaults to 'name'.
        direction (str): Sort direction, either 'ascending' or 'descending'. Defaults to 'ascending'.
    """
    validate_sketchbook_path(sketchbook_folder)
    sketch_files_list = get_sketchbook_files(sketchbook_folder)
    sketches = get_sketchbook_files_info(sketch_files_list, sketchbook_folder)
    sorted_sketches = sort_sketch_files(sketches, sort, direction)
    pretty_print_sketchbook(sorted_sketches, sketchbook_folder)


def get_sketchbook_files(sketchbook_folder: str = "./") -> List[str]:
    """
    Gets all sketch files from the sketchbook folder.

    Args:
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.

    Returns:
        List[str]: List of sketch file names.
    """
    files = os.listdir(sketchbook_folder)
    exclude = ["__init__.py", "__pycache__", ".DS_Store", "imgui.ini"]
    return [f for f in files if f not in exclude]


def get_sketchbook_files_info(
    sketches: List[str], sketchbook_folder: str = "./"
) -> List[Dict[str, Any]]:
    """
    Gets information about all sketch files in the sketchbook folder.

    Args:
        sketches (List[str]): List of sketch file names.
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.

    Returns:
        List[Dict[str, Any]]: List of sketch information dictionaries.
    """
    sketch_infos = []
    for sketch in sketches:
        sketch_info = get_sketch_info(sketch, sketchbook_folder)
        sketch_infos.append(sketch_info)
    return sketch_infos


def sort_sketch_files(
    sketch_files: List[Dict[str, Any]], sort: str = "name", direction: str = "ascending"
) -> List[Dict[str, Any]]:
    """
    Sorts sketch files by name, size, modified or created.

    Args:
        sort (str): Sort sketches by name, size, modified or created. Defaults to 'name'.
        sketch_files (List[Dict[str, Any]]): List of sketch information dictionaries.
        direction (str): Sort direction, either 'ascending' or 'descending'. Defaults to 'ascending'.

    Returns:
        List[Dict[str, Any]]: List of sorted sketch information dictionaries.
    """
    reverse = True if direction == "descending" else False
    if sort == "name":
        return sorted(sketch_files, key=lambda k: k["name"], reverse=reverse)
    elif sort == "size":
        return sorted(sketch_files, key=lambda k: k["size"], reverse=reverse)
    elif sort == "modified":
        return sorted(sketch_files, key=lambda k: k["modified"], reverse=reverse)
    elif sort == "created":
        return sorted(sketch_files, key=lambda k: k["created"], reverse=reverse)
    else:
        return sketch_files


def get_sketch_info(sketch_file: str, sketchbook_folder: str = "./") -> Dict[str, Any]:
    """
    Gets information about a specific sketch file.

    Args:
        sketch_file (str): Name of the sketch file.
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.

    Returns:
        Dict[str, Any]: Dictionary containing sketch information such as name, path, size, modified and created times.
    """
    validate_sketch_file(sketch_file, sketchbook_folder)
    datetimefmt = "%Y-%m-%d %H:%M:%S"
    if not sketch_file.endswith(".py"):
        sketch_file += ".py"
    file_path = os.path.join(sketchbook_folder, sketch_file)
    module_name = os.path.splitext(sketch_file)[0]
    file_info = os.stat(file_path)
    size = file_info.st_size
    modified = datetime.datetime.fromtimestamp(file_info.st_mtime).strftime(datetimefmt)
    created = datetime.datetime.fromtimestamp(file_info.st_ctime).strftime(datetimefmt)
    return {
        "name": module_name,
        "path": file_path,
        "size": size,
        "modified": modified,
        "created": created,
    }


def pretty_print_sketchbook(
    sketches: List[Dict[str, Any]], sketchbook_folder: str = "./"
) -> None:
    """
    Pretty prints the sketchbook information.

    Args:
        sketches (List[Dict[str, Any]]): List of sketch information dictionaries.
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.
    """
    validate_sketchbook_path(sketchbook_folder)
    print(f"\nSketchbook '{sketchbook_folder}':\n")
    print(
        f"  {'Index':<5} {'Sketch':<25} {'Size':<10} {'Modified':<20} {'Created':<20}"
    )
    print(f"  {'-'*5:<5} {'-'*25:<25} {'-'*10:<10} {'-'*20:<20} {'-'*20:<20}")
    for i, sketch in enumerate(sketches):
        print(
            f"  {i:<5} {sketch['name']:<25} {sketch['size']:<10} {sketch['modified']:<20} {sketch['created']:<20}"
        )
    print()


def run_sketch_by_index(
    index: int, sketchbook_folder: str = "./", *args: Any, **kwargs: Any
) -> None:
    """
    Runs a sketch by its index in the sketchbook.

    Args:
        index (int): Index of the sketch to run.
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.
    """
    validate_sketchbook_path(sketchbook_folder)
    files = get_sketchbook_files(sketchbook_folder)
    for file in files:
        if file.endswith(".py"):
            module_name = os.path.splitext(file)[0]
            file_path = os.path.join(sketchbook_folder, file)
            if str(index) == module_name:
                try_import_and_run_sketch(module_name, file_path, *args, **kwargs)


def run_sketch_by_name(
    sketch_file: str, sketchbook_folder: str = "./", *args: Any, **kwargs: Any
) -> None:
    """
    Runs a sketch by its file name.

    Args:
        sketch_file (str): Name of the sketch file.
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.
    """
    validate_sketchbook_path(sketchbook_folder)
    if not sketch_file.endswith(".py"):
        sketch_file += ".py"
    file_path = os.path.join(sketchbook_folder, sketch_file)
    module_name = os.path.splitext(sketch_file)[0]
    try_import_and_run_sketch(module_name, file_path, *args, **kwargs)


def run_random_sketch(sketchbook="./"):
    """
    Runs a random sketch from the sketchbook.

    Args:
        sketchbook (str): Path to the sketchbook folder. Defaults to current directory.
    """
    validate_sketchbook_path(sketchbook)
    files = get_sketchbook_files(sketchbook)
    sketch_file = files[random.randint(0, len(files) - 1)]
    run_sketch_by_name(sketch_file, sketchbook)


def try_import_and_run_sketch(
    module_name: str, file_path: str, *args: Any, **kwargs: Any
) -> None:
    """
    Tries to import and run a sketch from a given file.

    Args:
        module_name (str): Name of the module.
        file_path (str): Path to the file containing the module.
    """
    try:
        module = import_sketch(module_name, file_path)
        run_sketch_function_from_module(module, "sketch", file_path, *args, **kwargs)
    except Exception as e:
        print(f"Error running {module_name}: {str(e)}")


def import_sketch(module_name: str, file_path: str) -> Any:
    """
    Imports a sketch from a given file.

    Args:
        module_name (str): Name of the module.
        file_path (str): Path to the file containing the module.

    Returns:
        Any: Imported module.
    """
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None

    if not module_name:
        print("Module name is empty or invalid.")
        return None
    try:
        print(f"Importing {module_name} from {file_path}...")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        error_type = type(e).__name__
        print(f"Error importing {module_name} ({error_type}): {str(e)}")
        return None


def run_sketch_function_from_module(
    module: Any, function_name: str, file_path: str, *args: Any, **kwargs: Any
) -> None:
    """
    Runs a specific function from a given module.

    Args:
        module (Any): The imported module.
        function_name (str): Name of the function to run.
        file_path (str): Path to the file containing the module.
    """
    try:
        if hasattr(module, function_name) and callable(getattr(module, function_name)):
            print(f"Running {function_name} from {file_path}...")
            getattr(module, function_name)(*args, **kwargs)
        else:
            print(f"{module} does not have a '{function_name}' function.")
    except Exception as e:
        print(f"Error running {function_name} from {file_path}: {str(e)}")


def validate_sketchbook_path(sketchbook_folder: str = "./") -> None:
    """
    Validates if the given sketchbook folder exists.

    Args:
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.

    Raises:
        SystemExit: If the sketchbook folder does not exist.
    """
    if not os.path.isdir(sketchbook_folder):
        print(f"Sketchbook folder '{sketchbook_folder}' does not exist.")
        sys.exit(1)


def validate_sketch_file(sketch_file: str, sketchbook_folder: str = "./") -> None:
    """
    Validates if the given sketch file exists in the sketchbook folder.

    Args:
        sketch_file (str): Name of the sketch file.
        sketchbook_folder (str): Path to the sketchbook folder. Defaults to current directory.

    Raises:
        SystemExit: If the sketch file does not exist.
    """
    validate_sketchbook_path(sketchbook_folder)
    if not sketch_file.endswith(".py"):
        sketch_file += ".py"
    file_path = os.path.join(sketchbook_folder, sketch_file)
    if not os.path.isfile(file_path):
        print(f"Sketch file '{file_path}' does not exist.")
        sys.exit(1)


def main(*args, **kwargs):
    """
    Main function for running the sketchbook from the command line.
    """
    if "sketchbook" in kwargs:
        sketchbook = kwargs["sketchbook"]
    else:
        sketchbook = "./"
    if "sketch" in kwargs:
        sketch = kwargs["sketch"]
        if isinstance(sketch, str):
            run_sketch_by_name(sketch, sketchbook, *args, **kwargs)
        elif isinstance(sketch, int):
            print(f"Running sketch by index: {sketch}")
            run_sketch_by_index(sketch, sketchbook, *args, **kwargs)
    elif "sketches" in kwargs:
        sort = kwargs["sort"] if "sort" in kwargs else "name"
        direction = kwargs["direction"] if "direction" in kwargs else "ascending"
        list_sketches(sketchbook, sort, direction)
        exit()
    elif "random" in kwargs:
        run_random_sketch(sketchbook)
    else:
        list_sketches(sketchbook)
        exit()


if __name__ == "__main__":
    fire.Fire(main)
