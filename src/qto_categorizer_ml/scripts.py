"""Scripts for the CLI application."""

# %% IMPORTS

import argparse
import json
import sys

from qto_categorizer_ml import settings
from qto_categorizer_ml.io import configs

# %% PARSERS

parser = argparse.ArgumentParser(description="Run an AI/ML job from YAML/JSON configs.")
parser.add_argument("files", nargs="*", help="Config files for the job (local or S3 path).")
parser.add_argument("-e", "--extras", nargs="*", default=[], help="Config strings for the job.")
parser.add_argument("-s", "--schema", action="store_true", help="Print settings schema and exit.")


# %% SCRIPTS


def main(argv: list[str] | None = None) -> int:
    """Main script for the application."""
    args = parser.parse_args(argv)
    schema = settings.MainSettings.model_json_schema()
    if args.schema:
        schema = settings.MainSettings.model_json_schema()
        json.dump(schema, sys.stdout, indent=4)
        return 0  # success
    files = [configs.parse_file(file) for file in args.files]
    strings = [configs.parse_string(string) for string in args.extras]
    if len(files) == 0 and len(strings) == 0:
        raise RuntimeError("No configs provided.")
    config = configs.merge_configs([*files, *strings])
    object_ = configs.to_object(config)  # python object
    setting = settings.MainSettings.model_validate(object_)
    with setting.job as runner:
        runner.run()  # execute
        return 0  # success
