"""Command-line entries for the module."""

import click

import nima_io.read as ir


# TODO: test for version
@click.command()
@click.argument("fileA", type=click.Path(exists=True, dir_okay=False))
@click.argument("fileB", type=click.Path(exists=True, dir_okay=False))
@click.version_option()
def imgdiff(filea: str, fileb: str) -> None:
    """Compare two files (microscopy-data); first metadata then all pixels."""
    try:
        are_equal = ir.diff(filea, fileb)
        if are_equal:
            print("Files seem equal.")
        else:
            print("Files differ.")
    except Exception as read_problem:
        msg = f"Bioformats unable to read files. Exception: {read_problem}"
        raise SystemExit(msg) from read_problem


# if __name__ == "__main__":
# imgdiff(prog_name="imdff")  # pragma: no cover
