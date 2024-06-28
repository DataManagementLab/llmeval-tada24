import collections
import itertools
import logging
import pathlib
from typing import Any

import attrs
import cattrs
import pandas as pd
import tqdm
from sklearn.metrics import classification_report

from lib.data import dump_json

logger = logging.getLogger(__name__)


def extract_text_from_response(response: dict) -> str | None:
    """Extract the text from an OpenAI API response.

    Args:
        response: An OpenAI API response.

    Returns:
        The generated text or None if the API request failed.
    """
    if "choices" not in response.keys():
        return None

    return response["choices"][0]["message"]["content"]


def compute_table_sparsity(df: pd.DataFrame) -> float:
    """Compute the sparsity of the given table as the fraction of values that is nan.

    Args:
        df: The given table.

    Returns:
        The table sparsity.
y

    >>> compute_table_sparsity(pd.DataFrame({"a": [1, None], "b": [None, 2]}))
    0.5
    """
    return df.isna().sum().sum() / (len(df.index) * len(df.columns))


@attrs.define
class Accuracy:
    """Accuracy metric."""
    correct: int
    incorrect: int

    @property
    def total(self) -> int:
        """Total number of instances.

        Returns:
            The total number of instances.

        >>> Accuracy(1, 1).total
        2
        """
        return self.correct + self.incorrect

    @property
    def accuracy(self) -> float:
        """Accuracy score.

        Returns:
            The accuracy score.

        >>> Accuracy(1, 1).accuracy
        0.5
        """
        return self.correct / self.total

    @classmethod
    def empty(cls) -> "Accuracy":
        """Create an empty accuracy object.

        Returns:
            An empty accuracy object.

        >>> Accuracy.empty()
        Accuracy(correct=0, incorrect=0)
        """
        return cls(0, 0)

    def push(self, is_correct: bool) -> None:
        """Include the given instance in the accuracy.

        Args:
            is_correct: Whether the instance is correct.

        >>> acc = Accuracy(0, 0)
        >>> acc.push(True)
        >>> acc
        Accuracy(correct=1, incorrect=0)
        """
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1

    def __add__(self, other: "Accuracy") -> "Accuracy":
        return Accuracy(self.correct + other.correct, self.incorrect + other.incorrect)

    def __radd__(self, other: "Accuracy") -> None:
        self.correct += other.correct
        self.incorrect += other.incorrect

    def report(self):
        """Report the accuracy and relevant scores."""
        return {
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy
        }


@attrs.define
class ColumnTaskResults:
    """Results for a column-level task like column type inference."""
    num_columns_deviations: list[int]
    num_tables_with_column_at_idx: dict[int, int]

    classification_report: dict
    classification_report_by_idx: dict[int, dict]
    classification_report_by_data_type: dict[str, dict]
    classification_report_by_sparsity: dict[float, dict]
    classification_report_by_num_columns: dict[int, dict]

    missing_column_adjusted_classification_report: dict
    missing_column_adjusted_classification_report_by_idx: dict[int, dict]
    missing_column_adjusted_classification_report_by_data_type: dict[str, dict]
    missing_column_adjusted_classification_report_by_sparsity: dict[float, dict]
    missing_column_adjusted_classification_report_by_num_columns: dict[int, dict]

    not_even_a_column_type: list[str]

    @staticmethod
    def _pad_sequences(
            a: list[list[str]],
            b: list[list[str]],
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Pad the shorter lists by appending "MISSING".

        >>> x = ["a", "b", "c"]
        >>> y = ["a", "b"]
        >>> ColumnTaskResults._pad_sequences([x], [y])
        ([['a', 'b', 'c']], [['a', 'b', 'MISSING']])

        >>> x = ["a", "b"]
        >>> y = ["a", "b", "c"]
        >>> z = ["1", "2"]
        >>> ColumnTaskResults._pad_sequences([x, z], [y])
        ([['a', 'b', 'MISSING'], ['1', '2', 'MISSING']], [['a', 'b', 'c']])
        """
        assert len(set(map(len, a))) == 1
        assert len(set(map(len, b))) == 1

        if len(a[0]) < len(b[0]):
            diff = len(b[0]) - len(a[0])
            for ix in range(len(a)):
                a[ix] = a[ix] + ["MISSING"] * diff
        else:
            diff = len(a[0]) - len(b[0])
            for ix in range(len(b)):
                b[ix] = b[ix] + ["MISSING"] * diff

        return a, b

    @classmethod
    def _insert_and_pad_sequences(
            cls,
            a: list[list[str]],
            b: list[list[str]],
            indexes: list[int]
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Insert "MISSING" at the given indexes into the shorter lists, then pad them.

        >>> x = ["a", "b", "c", "d", "e"]
        >>> y = ["a", "b"]
        >>> z = ["1", "2"]
        >>> ColumnTaskResults._insert_and_pad_sequences([x], [y, z], [1])
        ([['a', 'b', 'c', 'd', 'e']], [['a', 'MISSING', 'b', 'MISSING', 'MISSING'], ['1', 'MISSING', '2', 'MISSING', 'MISSING']])
        """
        assert len(set(map(len, a))) == 1
        assert len(set(map(len, b))) == 1

        if len(a[0]) < len(b[0]):
            for ix in range(len(a)):
                a[ix] = a[ix].copy()
                for idx in indexes:
                    # noinspection PyTypeChecker
                    a[ix].insert(idx, "MISSING")
        else:
            for ix in range(len(b)):
                b[ix] = b[ix].copy()
                for idx in indexes:
                    # noinspection PyTypeChecker
                    b[ix].insert(idx, "MISSING")

        return cls._pad_sequences(a, b)

    @staticmethod
    def _classification_report(
            flat_true_values: list[str],
            flat_pred_values: list[str],
            labels: list[str]
    ) -> dict:
        return classification_report(
            [str(v) for v in flat_true_values],
            [str(v) for v in flat_pred_values],
            output_dict=True,
            zero_division=0.0,
            labels=labels
        )

    @classmethod
    def _classification_report_by(
            cls,
            flat_true_values_by: dict[Any, list[str]],
            flat_pred_values_by: dict[Any, list[str]],
            labels
    ) -> dict[Any, dict]:
        classification_reports_by = {}
        for by in flat_true_values_by.keys():
            classification_reports_by[by] = cls._classification_report(
                flat_true_values_by[by],
                flat_pred_values_by[by],
                labels
            )
        return classification_reports_by

    @classmethod
    def compute(
            cls,
            all_true_values: list[list[str | None]],
            # None is used for unspecified true values which are ignored in the evaluation
            all_pred_values: list[list[str]],
            all_data_types: list[list[str]],
            all_sparsities: list[list[float]],
            all_column_types: list[str],
            adjust_missing_columns_up_to: int,
            desc: str
    ) -> "ColumnTaskResults":
        """Compute the results for the given lists of true and predicted instances.

        Each instance must be a list of string values.

        Missing column adjustement inserts "MISSING" for missing columns such that the accuracy is maximized:

        * ["a", "b", "c"], ["a", "c", "MISSING"] ==> 1 correct, 2 incorrect
        * ["a", "b", "c"], ["a", "MISSING", "c"] ==> 2 correct, 1 incorrect


        Args:
            all_true_values: List of true instances.
            all_pred_values: List of predicted instances.
            all_data_types: List that assigns each value to a data type.
            all_sparsities: List that assigns each value a table sparsity.
            all_column_types: List of all possible column types.
            adjust_missing_columns_up_to: Up to how many missing columns should be adjusted for.
            desc: tqdm description.

        Returns:
            The evaluation results.
        """
        all_column_types_set = set(all_column_types)
        all_num_columns = [[len(true_values)] * len(true_values) for true_values in all_true_values]

        assert len(all_true_values) == len(all_pred_values)
        assert len(all_true_values) == len(all_data_types)
        assert len(all_true_values) == len(all_sparsities)
        assert len(all_true_values) == len(all_num_columns)
        assert all(len(a) == len(b) for a, b in zip(all_true_values, all_data_types))
        assert all(len(a) == len(b) for a, b in zip(all_true_values, all_sparsities))
        assert all(len(a) == len(b) for a, b in zip(all_true_values, all_num_columns))

        results = cls(
            num_columns_deviations=[],
            num_tables_with_column_at_idx=collections.Counter(),
            classification_report={},  # initialize later,
            classification_report_by_idx={},  # initialize_later
            classification_report_by_data_type={},  # initialize_later
            classification_report_by_sparsity={},  # initialize_later
            classification_report_by_num_columns={},  # initialize_later
            missing_column_adjusted_classification_report={},  # initialize_later
            missing_column_adjusted_classification_report_by_idx={},  # initialize_later
            missing_column_adjusted_classification_report_by_data_type={},  # initialize_later
            missing_column_adjusted_classification_report_by_sparsity={},  # initialize_later
            missing_column_adjusted_classification_report_by_num_columns={},  # initialize_later
            not_even_a_column_type=[]
        )

        # evaluation by padding sequences to the same length

        flat_padded_true_values, flat_padded_pred_values = [], []
        flat_padded_true_values_by_idx = collections.defaultdict(list)
        flat_padded_pred_values_by_idx = collections.defaultdict(list)
        flat_padded_true_values_by_data_type = collections.defaultdict(list)
        flat_padded_pred_values_by_data_type = collections.defaultdict(list)
        flat_padded_true_values_by_sparsity = collections.defaultdict(list)
        flat_padded_pred_values_by_sparsity = collections.defaultdict(list)
        flat_padded_true_values_by_num_columns = collections.defaultdict(list)
        flat_padded_pred_values_by_num_columns = collections.defaultdict(list)
        for inst_true_values, inst_pred_values, inst_data_types, inst_sparsities, inst_num_columns in tqdm.tqdm(
                zip(all_true_values, all_pred_values, all_data_types, all_sparsities, all_num_columns),
                desc=f"{desc} - padded sequences",
                total=len(all_true_values)
        ):
            logger.debug(f"TRUE: {inst_true_values}")
            logger.debug(f"PRED: {inst_pred_values}")

            results.num_columns_deviations.append(len(inst_pred_values) - len(inst_true_values))

            (inst_padded_true_values, inst_padded_data_types, inst_padded_sparsities, inst_padded_num_columns), (
                inst_padded_pred_values,) = cls._pad_sequences(
                [inst_true_values, inst_data_types, inst_sparsities, inst_num_columns],
                [inst_pred_values]
            )
            for idx, (
                    padded_true_value, padded_pred_value, padded_data_type, padded_sparsity,
                    padded_num_column) in enumerate(
                zip(inst_padded_true_values, inst_padded_pred_values, inst_padded_data_types,
                    inst_padded_sparsities, inst_padded_num_columns)):
                if padded_true_value is not None:  # ignore all columns for which the true value is None
                    flat_padded_true_values.append(padded_true_value)
                    flat_padded_pred_values.append(padded_pred_value)

                    if padded_pred_value not in all_column_types_set:
                        results.not_even_a_column_type.append(padded_pred_value)

                    if padded_true_value != "MISSING":
                        results.num_tables_with_column_at_idx[idx] += 1
                        flat_padded_true_values_by_idx[idx].append(padded_true_value)
                        flat_padded_pred_values_by_idx[idx].append(padded_pred_value)
                        flat_padded_true_values_by_data_type[padded_data_type].append(padded_true_value)
                        flat_padded_pred_values_by_data_type[padded_data_type].append(padded_pred_value)
                        flat_padded_true_values_by_sparsity[padded_sparsity].append(padded_true_value)
                        flat_padded_pred_values_by_sparsity[padded_sparsity].append(padded_pred_value)
                        flat_padded_true_values_by_num_columns[padded_num_column].append(padded_true_value)
                        flat_padded_pred_values_by_num_columns[padded_num_column].append(padded_pred_value)

        results.classification_report = cls._classification_report(
            flat_padded_true_values,
            flat_padded_pred_values,
            all_column_types
        )

        results.classification_report_by_data_type = cls._classification_report_by(
            flat_padded_true_values_by_data_type,
            flat_padded_pred_values_by_data_type,
            all_column_types
        )

        results.classification_report_by_sparsity = cls._classification_report_by(
            flat_padded_true_values_by_sparsity,
            flat_padded_pred_values_by_sparsity,
            all_column_types
        )

        results.classification_report_by_num_columns = cls._classification_report_by(
            flat_padded_true_values_by_num_columns,
            flat_padded_pred_values_by_num_columns,
            all_column_types
        )

        results.classification_report_by_idx = cls._classification_report_by(
            flat_padded_true_values_by_idx,
            flat_padded_pred_values_by_idx,
            all_column_types
        )

        # evaluation by adjusting sequences to the same length

        flat_adjusted_true_values, flat_adjusted_pred_values = [], []
        flat_adjusted_true_values_by_idx = collections.defaultdict(list)
        flat_adjusted_pred_values_by_idx = collections.defaultdict(list)
        flat_adjusted_true_values_by_data_type = collections.defaultdict(list)
        flat_adjusted_pred_values_by_data_type = collections.defaultdict(list)
        flat_adjusted_true_values_by_sparsity = collections.defaultdict(list)
        flat_adjusted_pred_values_by_sparsity = collections.defaultdict(list)
        flat_adjusted_true_values_by_num_columns = collections.defaultdict(list)
        flat_adjusted_pred_values_by_num_columns = collections.defaultdict(list)
        for inst_true_values, inst_pred_values, inst_data_types, inst_sparsities, inst_num_columns in tqdm.tqdm(
                zip(all_true_values, all_pred_values, all_data_types, all_sparsities, all_num_columns),
                desc=f"{desc} - adjusted sequences",
                total=len(all_true_values)
        ):
            logger.debug(f"TRUE: {inst_true_values}")
            logger.debug(f"PRED: {inst_pred_values}")

            inst_adjusted_true_values = None
            inst_adjusted_pred_values = None
            inst_adjusted_data_types = None
            inst_adjusted_sparsities = None
            inst_adjusted_num_columns = None

            add_num = abs(len(inst_pred_values) - len(inst_true_values))
            if add_num > adjust_missing_columns_up_to:
                logger.warning(
                    f"The difference in the number of columns ({add_num}) is greater than the configured maximum adjustment ({adjust_missing_columns_up_to})!")
                add_num = adjust_missing_columns_up_to

            if add_num == 0:
                (inst_adjusted_true_values, inst_adjusted_data_types, inst_adjusted_sparsities,
                 inst_adjusted_num_columns), (inst_adjusted_pred_values,) = cls._pad_sequences(
                    [inst_true_values, inst_data_types, inst_sparsities, inst_num_columns],
                    [inst_pred_values]
                )
            else:  # adjustment necessary
                best_acc = None
                for indexes in itertools.combinations(range(max(len(inst_pred_values), len(inst_true_values))),
                                                      r=add_num):
                    (tmp_adjusted_true_values, tmp_adjusted_data_types, tmp_adjusted_sparsities,
                     tmp_adjusted_num_columns), (tmp_adjusted_pred_values,) = cls._insert_and_pad_sequences(
                        [inst_true_values, inst_data_types, inst_sparsities, inst_num_columns],
                        [inst_pred_values],
                        indexes
                    )

                    acc = Accuracy.empty()
                    for true_value, pred_value in zip(tmp_adjusted_true_values, tmp_adjusted_pred_values):
                        acc.push(
                            true_value is None or true_value == pred_value)  # if the true value is None, it is always correct

                    if best_acc is None or acc.accuracy > best_acc.accuracy:
                        best_acc = acc
                        inst_adjusted_true_values = tmp_adjusted_true_values
                        inst_adjusted_pred_values = tmp_adjusted_pred_values
                        inst_adjusted_data_types = tmp_adjusted_data_types
                        inst_adjusted_sparsities = tmp_adjusted_sparsities
                        inst_adjusted_num_columns = tmp_adjusted_num_columns

            for idx, (adjusted_true_value, adjusted_pred_value, adjusted_data_type, adjusted_sparsity,
                      adjusted_num_column) in enumerate(
                zip(inst_adjusted_true_values, inst_adjusted_pred_values, inst_adjusted_data_types,
                    inst_adjusted_sparsities, inst_adjusted_num_columns)):
                if adjusted_true_value is not None:
                    flat_adjusted_true_values.append(adjusted_true_value)
                    flat_adjusted_pred_values.append(adjusted_pred_value)

                    if adjusted_true_value != "MISSING":
                        flat_adjusted_true_values_by_idx[idx].append(adjusted_true_value)
                        flat_adjusted_pred_values_by_idx[idx].append(adjusted_pred_value)
                        flat_adjusted_true_values_by_data_type[adjusted_data_type].append(adjusted_true_value)
                        flat_adjusted_pred_values_by_data_type[adjusted_data_type].append(adjusted_pred_value)
                        flat_adjusted_true_values_by_sparsity[adjusted_sparsity].append(adjusted_true_value)
                        flat_adjusted_pred_values_by_sparsity[adjusted_sparsity].append(adjusted_pred_value)
                        flat_adjusted_true_values_by_num_columns[adjusted_num_column].append(adjusted_true_value)
                        flat_adjusted_pred_values_by_num_columns[adjusted_num_column].append(adjusted_pred_value)

        results.missing_column_adjusted_classification_report = cls._classification_report(
            flat_adjusted_true_values,
            flat_adjusted_pred_values,
            all_column_types
        )

        results.missing_column_adjusted_classification_report_by_data_type = cls._classification_report_by(
            flat_adjusted_true_values_by_data_type,
            flat_adjusted_pred_values_by_data_type,
            all_column_types
        )

        results.missing_column_adjusted_classification_report_by_sparsity = cls._classification_report_by(
            flat_adjusted_true_values_by_sparsity,
            flat_adjusted_pred_values_by_sparsity,
            all_column_types
        )

        results.missing_column_adjusted_classification_report_by_num_columns = cls._classification_report_by(
            flat_adjusted_true_values_by_num_columns,
            flat_adjusted_pred_values_by_num_columns,
            all_column_types
        )

        results.missing_column_adjusted_classification_report_by_idx = cls._classification_report_by(
            flat_adjusted_true_values_by_idx,
            flat_adjusted_pred_values_by_idx,
            all_column_types
        )

        return results

    def save(self, path: pathlib.Path) -> None:
        """Save the results in the given directory.

        Args:
            path: Path at which to save the results.
        """
        dump_json(cattrs.unstructure(self), path)
