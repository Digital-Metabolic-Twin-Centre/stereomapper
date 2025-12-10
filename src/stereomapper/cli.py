"""Updated CLI with configuration management."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from stereomapper.config.loader import configure_from_cli
from stereomapper.config.settings import set_settings, get_settings
from stereomapper.config.integration import PipelineConfig
from stereomapper.domain.exceptions import ConfigurationError

from stereomapper.utils.logging import setup_logging
from stereomapper.domain.chemistry.validation import ChemistryValidator
from stereomapper.runners.pipeline import run_mol_distance_2D_batch


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the StereoMapper CLI."""
    parser = argparse.ArgumentParser(
        prog="stereomapper",
        description=(
            "Run StereoMapper: predict relationships between molecular "
            "structures for better cross-mapping."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the Stereomapper cross-mapping pipeline")

    mx = run_p.add_mutually_exclusive_group(required=True)
    mx.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Explicit list of input molfiles (file1.mol file2.sdf ...)",
    )
    mx.add_argument(
        "-d",
        "--input-dir",
        type=str,
        help=(
            "Directory containing input molfiles. Processes *.mol/*.sdf "
            "(optionally recursive)."
        ),
    )

    run_p.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="With --input-dir, search subdirectories recursively.",
    )
    run_p.add_argument(
        "-o",
        "--sqlite-output",
        required=True,
        help="Path to results SQLite DB (will be created if missing).",
    )
    run_p.add_argument(
        "-p",
        "--cache-path",
        help="Path to cache SQLite DB. If omitted, a default is chosen.",
    )
    run_p.add_argument(
        "-f",
        "--fresh-cache",
        action="store_true",
        help="Create a fresh cache DB at --cache-path (overwrite if exists).",
    )
    run_p.add_argument(
        "-n",
        "--namespace",
        default="default",
        type=str,
        help="Namespace tag for cache/output entries.",
    )
    run_p.add_argument(
        "--relate-with-cache",
        action="store_true",
        help="Relate new structures with those already in the cache.",
    )

    performance_group = run_p.add_argument_group("Performance Options")
    performance_group.add_argument(
        "--chunk-size",
        type=int,
        metavar="N",
        help="Number of files processed per chunk (default: 100,000).",
    )
    performance_group.add_argument(
        "--timeout",
        type=int,
        metavar="SECONDS",
        help="Timeout for individual operations (default: 300).",
    )

    output_group = run_p.add_argument_group("Output Options")
    output_group.add_argument(
        "--no-errors",
        action="store_true",
        help="Exclude error records from output.",
    )
    output_group.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Include detailed error information in output.",
    )

    debug_group = run_p.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging.",
    )
    debug_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without processing files.",
    )
    debug_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling.",
    )
    debug_group.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Write logs to file instead of console.",
    )

    return parser


def main() -> None:
    """Main entry point for the StereoMapper CLI."""
    args = build_parser().parse_args()

    if args.cmd != "run":
        logging.error("Unknown command: %s", args.cmd)
        sys.exit(2)

    try:
        settings = configure_from_cli(args)
        set_settings(settings)

        log_level = "DEBUG" if settings.debug_mode else "WARNING"
        logger, _ = setup_logging(
            log_dir=str(settings.logging.file_path.parent)
            if settings.logging.file_path
            else "./logs",
            console=settings.logging.console_output,
            level=log_level,
        )

        if settings.debug_mode:
            logger.info("Configuration loaded successfully")
            logger.debug("Configuration details:")
            config_dict = settings.to_dict()
            for section, values in config_dict.items():
                logger.debug("  %s: %s", section, values)

        inputs = _collect_input_files(args, logger)

        if args.input:
            settings.input_files = [Path(f) for f in inputs]
        else:
            settings.input_directory = Path(args.input_dir)

        out_path = Path(args.sqlite_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Processing Configuration:")
        logger.info("  Input files: %s", len(inputs))
        logger.info("  Chunk size: %s", f"{settings.processing.chunk_size:,}")
        logger.info("  Max workers: %s", settings.processing.max_workers)
        logger.info(
            "  Cache: %s",
            "Fresh" if settings.database.fresh_cache else "Existing",
        )
        logger.info("  Output: %s", out_path)

        if settings.dry_run:
            logger.info("DRY RUN MODE - configuration validated successfully")
            logger.info("Processing would proceed with the above configuration")
            _print_dry_run_summary(settings, inputs, out_path)
            sys.exit(0)

        cfg = PipelineConfig.from_settings(
            settings,
            inputs=inputs,
            output_path=str(out_path),
        )

        res = run_mol_distance_2D_batch(cfg)

        logger.info("Pipeline completed successfully!")
        logger.info("Results summary:")
        logger.info("  Processed: %s inputs", res.n_inputs)
        logger.info("  Session pairs: %s", res.n_session_pairs)
        logger.info("  Cross-cache pairs: %s", res.n_cross_pairs)
        logger.info("  Output: %s", res.output_path or args.sqlite_output)

        sys.exit(0)

    except ConfigurationError as e:
        logging.error("Configuration error: %s", e.message)
        if getattr(e, "suggestions", None):
            logging.error("Suggestions:")
            for suggestion in e.suggestions:
                logging.error("  - %s", suggestion)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        if get_settings().debug_mode:
            logging.exception("Full traceback:")
        sys.exit(1)


def _collect_input_files(args, logger) -> List[str]:
    """Collect input files from CLI arguments."""
    if args.input:
        inputs = []
        for item in args.input:
            p = Path(item)
            if p.is_file():
                inputs.append(str(p.resolve()))
            else:
                for ext in (".mol", ".sdf"):
                    matches = list(p.parent.glob(p.name)) if p.name else []
                    if matches:
                        inputs.extend(
                            str(m.resolve())
                            for m in matches
                            if m.suffix.lower() in (".mol", ".sdf")
                        )

        if not inputs:
            raise ConfigurationError(
                "No valid input files found in --input",
                config_field="input",
            ).add_suggestion(
                (
                    "Check that the specified files exist and use "
                    ".mol or .sdf extensions"
                )
            )

        return inputs

    dir_path = Path(args.input_dir)
    if not dir_path.is_dir():
        raise ConfigurationError(
            f"--input-dir {dir_path} is not a directory",
            config_field="input_dir",
        )

    inputs = _collect_inputs_from_dir(
        dir_path, recursive=args.recursive, logger=logger
    )
    if not inputs:
        msg = (
            f"No .mol/.sdf files found in {dir_path}"
            f"{'(recursive)' if args.recursive else ''}"
        )
        raise ConfigurationError(msg, config_field="input_dir").add_suggestion(
            "Check the directory path and file extensions"
        )

    return inputs


def _collect_inputs_from_dir(dir_path: Path, recursive: bool, logger) -> List[str]:
    """Collect .mol and .sdf files from a directory."""
    pattern = "**/*" if recursive else "*"
    files = []

    for ext in (".mol", ".sdf"):
        files.extend(dir_path.glob(f"{pattern}{ext}"))

    uniq_sorted = sorted(
        {str(p.resolve()) for p in files if p.is_file()}
    )

    filtered = []
    skipped_wildcard = 0
    skipped_errors = 0

    for f in uniq_sorted:
        try:
            if not ChemistryValidator.is_wildcard_molfile(f):
                filtered.append(f)
            else:
                skipped_wildcard += 1
                logger.debug("Skipping wildcard structure: %s", f)
        except Exception as e:
            skipped_errors += 1
            logger.warning("Error checking file %s: %s", f, e)

    if skipped_wildcard > 0:
        logger.info("Skipped %s wildcard molecules", skipped_wildcard)
    if skipped_errors > 0:
        logger.warning(
            "Skipped %s files due to validation errors", skipped_errors
        )

    return filtered


def _print_dry_run_summary(
    settings, inputs: List[str], output_path: Path
) -> None:
    """Print a summary for dry run mode."""
    print("\n" + "=" * 60)
    print("DRY RUN SUMMARY")
    print("=" * 60)
    print(f"Input files:        {len(inputs):,}")
    print(f"Output path:        {output_path}")
    print(f"Cache path:         {settings.database.cache_path or 'Default'}")
    print(f"Fresh cache:        {settings.database.fresh_cache}")
    print(f"Chunk size:         {settings.processing.chunk_size:,}")
    print(f"Max workers:        {settings.processing.max_workers}")
    print(f"Namespace:          {settings.namespace}")
    print(f"Relate with cache:  {settings.relate_with_cache}")
    print(f"Debug mode:         {settings.debug_mode}")
    print("=" * 60)

    if inputs:
        print("Example input files:")
        for i, path in enumerate(inputs[:5]):
            print(f"  {i + 1}. {path}")
        if len(inputs) > 5:
            print(f"  ... and {len(inputs) - 5} more")
    print()


if __name__ == "__main__":
    main()
