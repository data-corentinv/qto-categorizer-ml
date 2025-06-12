"""Split dataframes into subsets (e.g., train/valid/test)."""

# %% IMPORTS

import abc
import typing as T

import numpy as np
import numpy.typing as npt
import pydantic as pdt
import typing_extensions as TX
from sklearn.model_selection import train_test_split

from qto_categorizer_ml.core import schemas

# %% TYPES

Index = npt.NDArray[np.int64]
TrainTestIndex = tuple[Index, Index]
TrainTestSplits = T.Iterator[TrainTestIndex]

# %% SPLITTERS


class Splitter(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a splitter.

    Use splitters to split data in sets.
    e.g., split between a train/test subsets.

    # https://scikit-learn.org/stable/glossary.html#term-CV-splitter
    """

    KIND: str

    @abc.abstractmethod
    def split(
        self, inputs: schemas.Inputs, targets: schemas.Targets, groups: Index | None = None
    ) -> TrainTestSplits:
        """Split a dataframe into subsets.

        Args:
            inputs (schemas.Inputs): model inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.

        Returns:
            TrainTestSplits: iterator over the dataframe train/test splits.
        """

    @abc.abstractmethod
    def get_n_splits(
        self, inputs: schemas.Inputs, targets: schemas.Targets, groups: Index | None = None
    ) -> int:
        """Get the number of splits generated.

        Args:
            inputs (schemas.Inputs): models inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.

        Returns:
            int: number of splits generated.
        """


class TrainTestSplitter(Splitter):
    """Split a dataframe into a train and test set.

    Parameters:
        shuffle (bool): shuffle dataset before splitting it.
        test_size (int | float): number/ratio for the test set.
        random_state (int): random state for the splitter object.
    """

    KIND: T.Literal["TrainTestSplitter"] = "TrainTestSplitter"

    test_size: int | float = 0.2
    random_state: int = 42
    stratify: bool = True  # keep outputs distribution

    @TX.override
    def split(
        self, inputs: schemas.Inputs, targets: schemas.Targets, groups: Index | None = None
    ) -> TrainTestSplits:
        index = np.arange(len(inputs))  # return integer position

        train_index, test_index = train_test_split(
            index,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=targets if self.stratify else None,
        )
        yield train_index, test_index

    @TX.override
    def get_n_splits(
        self, inputs: schemas.Inputs, targets: schemas.Targets, groups: Index | None = None
    ) -> int:
        return 1


SplitterKind = TrainTestSplitter
