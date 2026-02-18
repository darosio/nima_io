"""Command-line entries for the module."""

import click

import nima_io.read as ir


@click.command()
@click.argument("file_a", type=click.Path(exists=True, dir_okay=False))
@click.argument("file_b", type=click.Path(exists=True, dir_okay=False))
@click.version_option(message="%(version)s")
def imgdiff(file_a: str, file_b: str) -> None:
    """Compare two files (microscopy-data); first metadata then all pixels."""
    try:
        are_equal = ir.diff(file_a, file_b)
        print("Files seem equal." if are_equal else "Files differ.")
    except Exception as read_problem:
        msg = f"Unable to read files. Exception: {read_problem}"
        raise SystemExit(msg) from read_problem
