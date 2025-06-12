# %% IMPORTS

import os

import mypy_boto3_s3.service_resource as s3
from omegaconf import OmegaConf
from qto_categorizer_ml.io import configs

# %% PARSERS


def test_parse_file__local(tmp_path: str) -> None:
    # given
    text = """
    a: 1
    b: True
    c: [3, 4]
    """
    path = os.path.join(tmp_path, "config.yml")
    with open(path, "w", encoding="utf-8") as writer:
        writer.write(text)
    # when
    config = configs.parse_file(path)
    # then
    assert config == {
        "a": 1,
        "b": True,
        "c": [3, 4],
    }, "Local file config should be loaded correctly!"


def test_parse_file__remote_s3(s3_conf_object: s3.Object) -> None:
    # given
    body = b"""
    a: 1
    b: True
    c: [3, 4]
    """
    s3_conf_object.put(Body=body)  # write config to object
    path = f"s3://{s3_conf_object.bucket_name}/{s3_conf_object.key}"
    # when
    config = configs.parse_file(path)
    # then
    assert config == {
        "a": 1,
        "b": True,
        "c": [3, 4],
    }, "Local file config should be loaded correctly!"


def test_parse_string() -> None:
    # given
    text = """{"a": 1, "b": 2, "data": [3, 4]}"""
    # when
    config = configs.parse_string(text)
    # then
    assert config == {
        "a": 1,
        "b": 2,
        "data": [3, 4],
    }, "String config should be loaded correctly!"


# %% MERGERS


def test_merge_configs() -> None:
    # given
    confs = [OmegaConf.create({"x": i, i: i}) for i in range(3)]
    # when
    config = configs.merge_configs(confs)
    # then
    assert config == {
        0: 0,
        1: 1,
        2: 2,
        "x": 2,
    }, "Configs should be merged correctly!"


# %% CONVERTERS


def test_to_object() -> None:
    # given
    values = {
        "a": 1,
        "b": True,
        "c": [3, 4],
    }
    config = OmegaConf.create(values)
    # when
    object_ = configs.to_object(config)
    # then
    assert object_ == values, "Object should be the same!"
    assert isinstance(object_, dict), "Object should be a dict!"
