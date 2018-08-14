#===============================================================================
# Imports
#===============================================================================
try:
    from perfecthash.cli import main as cli_main
except ImportError:
    import sys
    from os.path import (
        join,
        abspath,
        dirname,
        normpath,
    )
    pythonlib_dir = normpath(join(dirname(abspath(__file__)), '../python'))
    sys.path.insert(0, pythonlib_dir)
    import perfecthash.cli
    from perfecthash.cli import main as cli_main

#===============================================================================
# Main
#===============================================================================
if __name__ == '__main__':
    cli_main(program_name='perfecthash', library_name='perfecthash')

# vi:set ts=8 sw=4 sts=4 expandtab tw=80 syntax=python                         :
