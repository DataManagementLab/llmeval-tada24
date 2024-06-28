import logging
import random
from typing import Any, Union

logger = logging.getLogger(__name__)

_shuffle_instances_random = random.Random(803270735)


def shuffle_instances(
        instances: list[Any],
        *other_instances: list[Any]
) -> Union[list[Any], tuple[list[Any], ...]]:
    """Shuffles instances inplace and returns the shuffled instances.

    Args:
        instances: The list of instances.

    Returns:
        The shuffled list of instances.
    """
    if not other_instances:
        _shuffle_instances_random.shuffle(instances)
        return instances
    else:
        all_instances = [instances, *other_instances]
        for instance in all_instances:
            assert len(instance) == len(instances), "All instances must have the same length."
        indices = list(range(len(instances)))
        _shuffle_instances_random.shuffle(indices)
        return tuple([instance[i] for i in indices]
                     for instance in all_instances)
