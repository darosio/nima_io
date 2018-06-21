import io
import sys

import imgread.read as ir
from docopt import docopt
from imgread import __version__ as version



def imgdiff():
    """
Compares two files (microscopy-data); first metadata then all pixels.

Usage:
  imgdiff <fileA> <fileB>
  imgdiff -h | --help
  imgdiff --version

Options:
  -h --help     Show this screen.
  --version     Show version.
    """
    args = docopt(imgdiff.__doc__, version=version)
    ir.ensure_VM()
    try:
        f = io.StringIO()
        with ir.stdout_redirector(f):
            are_equal = ir.diff(args['<fileA>'], args['<fileB>'])
        out = f.getvalue()
        with open("bioformats.log", 'a') as f:
            f.write("\n\n"+str(args)+"\n")
            f.write(out)
        if are_equal:
            print("Files seem equal.")
        else:
            print("Files differ.")
    except Exception:
        # can be moved to read_wrap function?
        raise SystemExit("Bioformats unable to read files.")
        instance = sys.exc_info()[1]
        print(instance)
    finally:
        ir.release_VM()
