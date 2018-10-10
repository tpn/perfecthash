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
    OutPathInvariant,
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
        self.addresses_text = b'\n'.join(self.addresses)

        from .cli import INTERACTIVE
        if not INTERACTIVE:
            self.ostream.write(self.addresses_text.decode('utf-8'))


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

class ExtractSingleCfgTarget(InvariantAwareCommand):
    """
    TBD.
    """
    _verbose_ = True


    dll = None
    _dll = None
    class DllArg(PathInvariant):
        _help = "Path to the DLL (e.g. C:\\Windows\\System32\\ntdll.dll)"
        _mandatory = True

    base_output_dir = None
    _base_output_dir = None
    class BaseOutputDirArg(DirectoryInvariant):
        _help = ("Base output directory.")
        _mandatory = False

    def run(self):
        InvariantAwareCommand.run(self)
        conf = self.conf
        out = self._out

        from .util import mkdir

        if not self._base_output_dir:
            base = self.conf.base_output_dir
            mkdir(base)
            self.base_output_dir = base

        output_dir = self._base_output_dir

        from .dumpbin import Dumpbin

        dll = self._dll
        db = Dumpbin(dll)

        if not db.is_cf_instrumented:
            raise CommandError("%s is not CF instrumented." % dll)
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

class TrailingSlashesAlign(InvariantAwareCommand):
    """
    Finds all multi-line macros and aligns trailing slashes where necessary.
    """

    path = None
    class PathArg(PathInvariant):
        _help = "path of the file"

    def run(self):
        out = self._out
        options = self.options
        verbose = self._verbose

        path = options.path

        from .sourcefile import SourceFile

        source = SourceFile(path)
        orig_data = source.data
        orig_lines = source.lines

        defines = source.defines
        multiline_macro_defines = source.multiline_macro_defines

        if not multiline_macro_defines:
            return

        lines = source.lines
        dirty = False

        from .util import align_trailing_slashes

        for (name, macro) in multiline_macro_defines.items():
            old_lines = macro.lines
            new_lines = align_trailing_slashes(old_lines)
            old_length = len(old_lines)
            new_length = len(new_lines)
            assert old_length == new_length, (old_length, new_length)
            if new_lines == old_lines:
                continue

            lines[macro.first_lineno:macro.last_lineno+1] = new_lines
            dirty = True
            out("Aligned trailing slashes for %s macro." % name)

        if dirty:
            text = '%s\n' % '\n'.join(lines)
            text = text.encode('utf-8')
            with open(path, 'wb') as f:
                f.write(text)

class UpdateRawCStringFile(InvariantAwareCommand):
    """
    Converts a file into a C char * format that can be #included by other C
    files.
    """
    _shortname_ = 'uf'

    input_path = None
    _input_path = None
    class InputPathArg(PathInvariant):
        _help = "path of input file"

    def run(self):
        out = self._out
        options = self.options
        verbose = self._verbose

        input_path = self._input_path

        from os.path import basename, dirname
        from .path import join_path

        base = basename(input_path)
        ix = base.rfind('.')
        assert ix != -1
        name = base[:ix]

        output_dir = dirname(input_path).replace('CompiledPerfectHashTable',
                                                 'PerfectHash')

        new_name = '%s_RawCString.h' % name

        output_path = join_path(output_dir, new_name)

        from .sourcefile import SourceFile

        input_source = SourceFile(input_path)

        source_lines = input_source.lines

        decl_lines = [
            '//',
            '// Auto-generated from ../CompiledPerfectHashTable/%s.' % base,
            '//',
            '',
            'DECLSPEC_ALIGN(16)',
            'const CHAR %sRawCStr[] =' % name,
        ]

        cstr_lines = [ '    %s' % l for l in input_source.lines_as_cstr() ]

        end_lines = [
            ';',
            '',
            'const STRING %sRawString = {' % name,
            '    sizeof(%sRawCStr) - sizeof(CHAR),' % name,
            '    sizeof(%sRawCStr),' % name,
            '#ifdef _WIN64',
            '    0,',
            '#endif',
            '    (PCHAR)&%sRawCStr,' % name,
            '};',
            '',
        ]

        lines = decl_lines + cstr_lines + end_lines

        text = '\n'.join(lines).encode('utf-8')

        actual = None

        try:
            with open(output_path, 'rb') as f:
                actual = f.read()
        except:
            pass

        if actual and actual == text:
            return

        with open(output_path, 'wb') as f:
            f.write(text)

        out("Updated ..\\PerfectHash\\%s." % basename(output_path))

class ReplaceUuid(InvariantAwareCommand):
    """
    Replaces UUIDs in a file with new ones.
    """

    path = None
    _path = None
    class PathArg(PathInvariant):
        _help = "path of file to replace UUIDs in"

    def run(self):
        out = self._out
        options = self.options
        verbose = self._verbose

        path = self._path

        import re
        from uuid import uuid4
        import linecache

        regex = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){4}[0-9a-f]{8}', re.I)

        with open(path, 'r') as f:
            text = f.read()

        uuid_map = {
            src_uuid: str(uuid4()).upper()
                for src_uuid in regex.findall(text)
        }

        for (old_uuid, new_uuid) in uuid_map.items():
            out("Replacing %s -> %s." % (old_uuid, new_uuid))
            text = text.replace(old_uuid, new_uuid)

        with open(path, 'w') as f:
            f.write(text)

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
