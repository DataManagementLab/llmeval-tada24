import json
import logging
import os
import pathlib
import shutil

logger = logging.getLogger(__name__)


def get_data_path() -> pathlib.Path:
    """Get the absolute path of the data directory.

    Returns:
        A pathlib.Path to the data directory.
    """
    path = pathlib.Path(os.path.dirname(__file__)).resolve() / ".." / "data/"
    os.makedirs(path, exist_ok=True)
    return path


def _prepare_directory(
        exp_name: str | None,
        dir_name: str,
        task_name: str,
        dataset_name: str,
        clear: bool
) -> pathlib.Path:
    if exp_name is None:
        path = get_data_path() / task_name / dataset_name / dir_name
    else:
        path = get_data_path() / task_name / dataset_name / "experiments" / exp_name / dir_name
    if path.is_dir() and clear:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def get_task_dir(task_name: str) -> pathlib.Path:
    """Directory in which task-specific data should be stored.

    Args:
        task_name: The name of the task.

    Returns:
        A pathlib.Path to the directory.
    """

    path = get_data_path() / task_name
    if not path.is_dir():
        os.makedirs(path, exist_ok=True)
    return path


def get_download_dir(task_name: str, dataset_name: str, clear: bool = False) -> pathlib.Path:
    """Directory in which to place the **downloaded data**.

    This path is the same for all experiments.

    Args:
        task_name: The name of the task.
        dataset_name: The name of the dataset.
        clear: Whether to clear the directory.

    Returns:
        A pathlib.Path to the directory.
    """
    return _prepare_directory(None, "download", task_name, dataset_name, clear)


def get_instances_dir(task_name: str, dataset_name: str, exp_name: str, clear: bool = False) -> pathlib.Path:
    """Directory in which to place the **preprocessed instances**.

    Args:
        task_name: The name of the task.
        dataset_name: The name of the dataset.
        exp_name: The name of the current experiment.
        clear: Whether to clear the directory.

    Returns:
        A pathlib.Path to the directory.
    """
    return _prepare_directory(exp_name, "instances", task_name, dataset_name, clear)


def get_requests_dir(task_name: str, dataset_name: str, exp_name: str, clear: bool = False) -> pathlib.Path:
    """Directory in which to place the **requests**.

    Args:
        task_name: The name of the task.
        dataset_name: The name of the dataset.
        exp_name: The name of the current experiment.
        clear: Whether to clear the directory.

    Returns:
        A pathlib.Path to the directory.
    """
    return _prepare_directory(exp_name, "requests", task_name, dataset_name, clear)


def get_responses_dir(task_name: str, dataset_name: str, exp_name: str, clear: bool = False) -> pathlib.Path:
    """Directory in which to place the **responses**.

    Args:
        task_name: The name of the task.
        dataset_name: The name of the dataset.
        exp_name: The name of the current experiment.
        clear: Whether to clear the directory.

    Returns:
        A pathlib.Path to the directory.
    """
    return _prepare_directory(exp_name, "responses", task_name, dataset_name, clear)


def get_results_dir(task_name: str, dataset_name: str, exp_name: str, clear: bool = False) -> pathlib.Path:
    """Directory in which to place the **results**.

    Args:
        task_name: The name of the task.
        dataset_name: The name of the dataset.
        exp_name: The name of the current experiment.
        clear: Whether to clear the directory.

    Returns:
        A pathlib.Path to the directory.
    """
    return _prepare_directory(exp_name, "results", task_name, dataset_name, clear)


def load_json(path: pathlib.Path) -> dict | list:
    """Load the JSON object from the given file path.

    Args:
        path: The pathlib.Path to the JSON file.

    Returns:
        The JSON object.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(obj: dict | list, path: pathlib.Path) -> None:
    """Dump the given JSON object to the given file path.

    Args:
        obj: The JSON object.
        path: The pathlib.Path to the JSON file.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file)


def load_str(path: pathlib.Path) -> str:
    """Load the string from the given file path.

    Args:
        path: The pathlib.Path to the TXT file.

    Returns:
        The string.
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def dump_str(s: str, path: pathlib.Path) -> None:
    """Dump the given string to the given file path.

    Args:
        s: The string.
        path: The pathlib.Path to the TXT file.
    """
    with open(path, "w", encoding="utf-8") as file:
        file.write(s)
