import json
import logging
from typing import Literal, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def linearize_table(
        table: pd.DataFrame,
        table_name: str | None,
        *,
        template: str,
        mode: Literal["csv"],
        csv_params: dict,
        markdown_params: dict,
        replace_na: Optional[str] = None
) -> str:
    """Linearize the given table.

    Template variables: {{table_name}}, {{table}}, {{newline}}

    Args:
        table: The table to linearize.
        table_name: The name of the table.
        template: The linearization template.
        mode: The linearization mode.
        csv_params: The parameters for pandas' to_csv method.
        markdown_params: The parameters for pandas' to_markdown method.

    Returns:
        The linearized table string.
    """
    if replace_na is not None:
        mask = table.isna()
        table = table.where(~mask, replace_na)
    template = template.replace("{{newline}}", "\n")
    if table_name:
        template = template.replace("{{table_name}}", table_name)

    if mode == "csv":
        table_linearized = table.to_csv(**csv_params)
    elif mode == "markdown":
        table_linearized = table.to_markdown(**markdown_params)
    elif mode == "key_value":
        table_linearized = "\n".join(", ".join(f"{k}: {v}" for k, v in zip(table.columns, row))
                                     for row in table.itertuples(index=False))
    else:
        raise AssertionError(f"Unsupported table serialization mode '{mode}'!")
    template = template.replace("{{table}}", table_linearized)

    return template


def linearize_list(
        l: list[Any],
        *,
        mode: Literal["csv"] | Literal["json_list"],
        sep: str,
        strip: bool
) -> str:
    """Linearize the given list.

    Args:
        l: The list to linearize.
        mode: The linearization mode.
        sep: The separator string.
        strip: Whether to strip whitespaces when delinearizing.

    Returns:
        The linearized list string.
    """
    if mode == "csv":
        return sep.join(l)
    elif mode == "json_list":
        return json.dumps(l)
    else:
        raise AssertionError(f"Unknown list serialization mode '{mode}'!")


def delinearize_list(
        s: str,
        *,
        mode: Literal["csv"] | Literal["json_list"],
        sep: str,
        strip: bool
) -> list[str] | None:
    """Delinearize the given string into a list.

    Args:
        s: The string to delinearize.
        mode: The linearization mode.
        sep: The separator string.
        strip: Whether to strip whitespaces when delinearizing.

    Returns:
        The delinearized list.
    """
    if mode == "csv":
        l = s.split(sep)
        if strip:
            l = [s.strip() for s in l]
        return l
    elif mode == "json_list":
        try:
            l = json.loads(s)
        except:
            try:
                s_new = f"{s}\"]"
                l = json.loads(s_new)
            except:
                try:
                    s_new = f"{s}]"
                    l = json.loads(s_new)
                except:
                    try:
                        s_new = f"{s}\"\"]"
                        l = json.loads(s_new)
                    except:
                        logger.warning(f"Delinearization failed, not JSON-parsable: '{s}'")
                        return None
        if not isinstance(l, list):
            logger.warning(f"Delinearization failed, not a list: '{s}'")
            return None
        l = [str(s) for s in l]
        if strip:
            l = [s.strip() for s in l]
        return l
    else:
        raise AssertionError(f"Unknown list serialization mode '{mode}'!")
