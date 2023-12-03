"""Command-line entries for the module."""
import io

import click

import nima_io.read as ir


# TODO: test for version
@click.command()
@click.argument("fileA", type=click.Path(exists=True, dir_okay=False))
@click.argument("fileB", type=click.Path(exists=True, dir_okay=False))
@click.version_option()
def imgdiff(filea: str, fileb: str) -> None:
    """Compare two files (microscopy-data); first metadata then all pixels."""
    ir.ensure_vm()

    try:
        f = io.StringIO()
        with ir.stdout_redirector(f):
            are_equal = ir.diff(filea, fileb)
        out = f.getvalue()
        with open("bioformats.log", "a") as log_file:
            log_file.write("\n\n" + f"{' '.join(['imgdiff', filea, fileb])}\n")
            log_file.write(out)

        if are_equal:
            print("Files seem equal.")
        else:
            print("Files differ.")
    except Exception as read_problem:
        msg = f"Bioformats unable to read files. Exception: {read_problem}"
        raise SystemExit(msg) from read_problem
    finally:
        ir.release_vm()


# if __name__ == "__main__":
# imgdiff(prog_name="imdff")  # pragma: no cover
