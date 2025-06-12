# %% IMPORTS

import os

from qto_categorizer_ml.core import schemas
from qto_categorizer_ml.io import datasets

# %% READERS


def test_csv_reader__local(inputs_path: str) -> None:
    # given
    reader = datasets.CSVReader(path=inputs_path)
    # when
    data = reader.read()
    # then
    assert data.ndim == 2, "Data should be a dataframe!"


# %% WRITERS


def test_csv_writer__local(targets: schemas.Targets, tmp_outputs_path: str) -> None:
    # given
    writer = datasets.CSVWriter(path=tmp_outputs_path)
    # when
    writer.write(data=targets)
    # then
    assert os.path.exists(tmp_outputs_path), "Data should be written!"
