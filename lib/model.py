import logging

import tiktoken

from lib.openai import openai_execute

logger = logging.getLogger(__name__)


def get_num_tokens(
        text: str,
        api_name: str,
        model: str
) -> int:
    """Compute the number of tokens of the text.

    Args:
        text: A given text.
        api_name: The name of the API to use.
        model: The name of the model to use.

    Returns:
        The number of tokens in the text.
    """
    if api_name == "openai" or api_name == "sapllmproxy" or api_name == "aicore":
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    else:
        raise AssertionError(f"Unknown API name '{api_name}'!")


def execute_requests_against_api(
        requests: list[dict],
        api_name: str
) -> list[dict]:
    """Execute a list of requests against one of the APIs.

    Args:
        requests: A list of API requests.
        api_name: The name of the API.

    Returns:
        A list of API responses.
    """
    if api_name == "openai":
        return openai_execute(requests, force=0.000000001)
    elif api_name == "aicore":
        from lib.aicore import aicore_execute
        return aicore_execute(requests, force=0.000000001)
    else:
        raise AssertionError(f"Unknown API name '{api_name}'!")
