# %% IMPORTS

import json

import pytest
from _pytest import capture as pc
from qto_categorizer_ml import scripts

# %% SCRIPTS


def test_schema(capsys: pc.CaptureFixture[str]) -> None:
    # given
    args = ["prog", "--schema"]
    # when
    scripts.main(args)
    capture = capsys.readouterr()
    # then
    assert capture.err == "", "Captured error should be empty!"
    assert json.loads(capture.out), "Captured output should be a JSON!"

def test_main__no_configs() -> None:
    # given
    argv: list[str] = []
    # when
    with pytest.raises(RuntimeError) as error:
        scripts.main(argv)
    # then
    assert error.match("No configs provided."), "RuntimeError should be raised!"
