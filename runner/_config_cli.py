import argparse
import os
import sys


CONFIG_ENV_VAR = "DEEPEYE_CONFIG_PATH"


def configure_from_cli(argv: list[str] | None = None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the TOML config file. Defaults to config/config.toml.",
    )
    args, remaining = parser.parse_known_args(argv)
    if args.config:
        os.environ[CONFIG_ENV_VAR] = args.config
    sys.argv = [sys.argv[0], *remaining]
    return args.config
