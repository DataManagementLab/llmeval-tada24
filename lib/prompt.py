import logging
import pathlib
import random
import re
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import pandas as pd

from lib.model import get_num_tokens

logger = logging.getLogger(__name__)

_sample_examples_random = random.Random(613907351)


def sample_examples(
        instance_path: pathlib.Path,
        instance_paths: list[pathlib.Path],
        *,
        num_examples: int
) -> list[pathlib.Path]:
    """Sample instances paths from all other instance paths.

    Args:
        instance_path: The path of the current instance.
        instance_paths: All instance paths.
        num_examples: The number of examples.

    Returns:
        A list of instance paths.
    """
    instance_paths = instance_paths.copy()
    instance_paths.remove(instance_path)
    return _sample_examples_random.sample(instance_paths, k=num_examples)


_sample_rows_random = np.random.default_rng(seed=964183484)


def sample_rows(
        table: pd.DataFrame,
        *other_tables: pd.DataFrame,
        num_rows: int
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """Sample rows from a pd.DataFrame.

    Args:
        table: The table to sample from.
        num_rows: The number of rows to sample.

    Returns:
        A pd.DataFrame with the sampled rows.
    """
    num_rows = min(num_rows, len(table.index))
    if not other_tables:
        return table.sample(n=num_rows, axis=0, random_state=_sample_rows_random,
                            ignore_index=True)  # axis=0 means sample rows
    else:
        ids = _sample_rows_random.choice(len(table.index), num_rows, replace=False)
        return tuple(other_table.iloc[ids] for other_table in (table,) + other_tables)


def max_tokens_for_ground_truth(ground_truth: str, api_name: str, model: str,
                                max_tokens_over_ground_truth: int | None) -> int | None:
    """Compute max_tokens based on the length of the ground truth and cfg.max_tokens_over_ground_truth.

    Args:
        ground_truth: The ground truth string.
        api_name: The name of the API.
        model: The model name.
        max_tokens_over_ground_truth: How many additional tokens should be allowed.

    Returns:
        The value for max_tokens.
    """
    ground_truth_len = get_num_tokens(ground_truth, api_name, model)
    return None if max_tokens_over_ground_truth is None else (ground_truth_len + max_tokens_over_ground_truth)


def fill_chat_template(
        template: list[dict[str, str] | str],
        **args
) -> list[dict[str, str]]:
    """Replace {{variables}} in the template with the given values.

    A variable can be a list of messages, a message, or a string.

    Warns in case of missing values, but not in case of unneeded values.

    Args:
        template: List of template messages containing {{variables}}.
        **args: The given string or message values for the variables.

    Returns:
        The filled-out template.
    >>> fill_chat_template([{"role": "user", "content": "My name is {{name}}."}, "{{greeting}}"], name="Micha", greeting={"role": "assistant", "content": "Nice to meet you!"})
    [{'role': 'user', 'content': 'My name is Micha.'}, {'role': 'assistant', 'content': 'Nice to meet you!'}]
    """
    template = deepcopy(template)
    # replace variables with values
    new_template = []
    for message in template:
        if isinstance(message, str):
            for key, value in args.items():
                template_key = "{{" + key + "}}"
                if template_key == message:
                    if isinstance(value, list):
                        new_template += value
                    elif isinstance(value, dict):
                        new_template.append(value)
                    else:
                        raise TypeError(
                            f"Value for key '{key}' must be a message dictionary or list of message dictionaries!")
                    break
            else:
                raise AssertionError(f"Missing value for template message variable '{message}'!")
        elif isinstance(message, dict):
            for key, value in args.items():
                template_key = "{{" + key + "}}"
                if template_key in message["content"]:
                    message["content"] = message["content"].replace(template_key, value)
            new_template.append(message)
        else:
            raise TypeError(f"Invalid type {type(message)} for template message!")

    # check that all variables have been filled
    for message in new_template:
        # noinspection RegExpRedundantEscape
        missing_keys = re.findall(r"\{\{(.*)\}\}", message["content"])
        if len(missing_keys) > 0:
            logger.warning(f"Missing values for template string variables {missing_keys}!")

    return new_template
