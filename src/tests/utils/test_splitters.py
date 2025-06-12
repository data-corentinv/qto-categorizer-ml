# %% IMPORTS

from qto_categorizer_ml.core import schemas
from qto_categorizer_ml.utils import splitters

# %% SPLITTERS


def test_train_test_splitter(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    # given
    test_size = 5
    random_state = 0
    stratify = False
    splitter = splitters.TrainTestSplitter(
        stratify=stratify, test_size=test_size, random_state=random_state
    )
    # when
    n_splits = splitter.get_n_splits(inputs=inputs, targets=targets)
    splits = list(splitter.split(inputs=inputs, targets=targets))
    train_index, test_index = splits[0]  # train/test indexes
    # then
    assert n_splits == len(splits) == 1, "Splitter should return 1 split!"
    assert len(test_index) == test_size, "Test index should have the given size!"
    assert (
        len(train_index) == len(targets) - test_size
    ), "Train index should have the remaining size!"
    assert not inputs.iloc[test_index].empty, "Test index should be a subset of the inputs!"
    assert not targets.iloc[train_index].empty, "Train index should be a subset of the targets!"
