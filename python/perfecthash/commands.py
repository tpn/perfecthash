#===============================================================================
# Imports
#===============================================================================
import sys

import textwrap

from .util import strip_linesep_if_present

from .command import (
    Command,
    CommandError,
)

from .invariant import (
    BoolInvariant,
    PathInvariant,
    StringInvariant,
    DirectoryInvariant,
    PositiveIntegerInvariant,
)

from .commandinvariant import (
    InvariantAwareCommand,
)

#===============================================================================
# Commands
#===============================================================================

class DocTest(InvariantAwareCommand):
    def run(self):
        quiet = self.options.quiet
        verbose = not quiet
        import doctest

        import perfecthash.dumpbin
        doctest.testmod(
            perfecthash.dumpbin,
            verbose=verbose,
            raise_on_error=True
        )

class DumpCfgTargets(InvariantAwareCommand):
    """
    Dumps a normalized guard control flow target address table, suitable for
    saving to a .txt file that can then be converted into a .keys file.  This
    works by parsing the output of `dumpbin /headers /loadconfig <dll>`.
    """
    _verbose_ = True

    dll = None
    _dll = None
    class DllArg(PathInvariant):
        _help = "Path to the DLL (e.g. C:\\Windows\\System32\\ntdll.dll)"
        _mandatory = True


    def run(self):
        InvariantAwareCommand.run(self)
        conf = self.conf
        out = self._out

        dll = self._dll

        from .dumpbin import Dumpbin

        db = Dumpbin(path=dll)

        if not db.is_cf_instrumented:
            raise CommandError("%s is not CF instrumented." % dll)

        self.addresses = db.guard_cf_func_table_addresses_base0
        self.addresses_text = '\n'.join(self.addresses)

        from .cli import INTERACTIVE
        if not INTERACTIVE:
            self.ostream.write(self.addresses_text)


class ExtractAllCfgTargets(InvariantAwareCommand):
    """
    TBD.
    """
    _verbose_ = True

    input_dir = None
    _input_dir = None
    class InputDirArg(DirectoryInvariant):
        _default = "C:\\Windows\\System32"
        _help = (
            "Directory to search for .dll, .sys and .exe files. "
            "[default: %default]"
        )

    base_output_dir = None
    _base_output_dir = None
    class BaseOutputDirArg(DirectoryInvariant):
        _help = ("Base output directory.")
        _mandatory = False

    def run(self):
        InvariantAwareCommand.run(self)
        conf = self.conf
        out = self._out

        from os import listdir
        from .util import mkdir
        from .path import join

        suffixes = ('.dll', '.exe', '.sys')

        if not self._base_output_dir:
            base = self.conf.base_output_dir
            mkdir(base)
            self.base_output_dir = base

        base = self._input_dir
        paths = [ join(base, p) for p in listdir(base) if p.endswith(suffixes) ]

        output_dir = self._base_output_dir

        from .dumpbin import Dumpbin

        from tqdm import tqdm

        for path in tqdm(paths):
            db = Dumpbin(path)
            if not db.is_cf_instrumented:
                continue
            db.save(output_dir)

class ComScratch(InvariantAwareCommand):
    """
    TBD.
    """
    _verbose_ = True

    def run(self):
        InvariantAwareCommand.run(self)
        conf = self.conf
        out = self._out

        from .wintypes import (
            HRESULT,
            ULONG,
            WINFUNCTYPE,
        )

        from perfecthash.dll import PerfectHash as phl

        PICLASSFACTORY = phl.PICLASSFACTORY
        PERFECT_HASH_KEYS = phl.PERFECT_HASH_KEYS
        PERFECT_HASH_KEYS = phl.PERFECT_HASH_KEYS

        cf = phl.get_class_factory()

        prototype = WINFUNCTYPE(ULONG, PICLASSFACTORY)
        add_ref = prototype(1, 'AddRef')

        import IPython
        IPython.embed()

        #result = add_ref(cf)

class LoadKeys(InvariantAwareCommand):
    """
    Loads keys.
    """
    _verbose_ = True

    path = None
    _path = None
    class PathArg(PathInvariant):
        _help = "Path to the keys file."
        _mandatory = True


    def run(self):
        InvariantAwareCommand.run(self)
        conf = self.conf
        out = self._out

        from perfecthash.dll import PerfectHash as phl

        keys = phl.Keys(self._path)
        bitmap = bin(keys.get_bitmap())[2:].zfill(32)
        ones = bitmap.count('1')

        self._out("Bitmap: %s [%d]" % (bitmap, ones))


# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
