#===============================================================================
# Imports
#===============================================================================

import re
import sys

import textwrap

from collections import namedtuple

from .util import (
    memoize,
    align_trailing_slashes,
    strip_linesep_if_present,
)

from .config import (
    get_or_create_config,
)

from .command import (
    Command,
    CommandError,
)

from .invariant import (
    BoolInvariant,
    PathInvariant,
    StringInvariant,
    DirectoryInvariant,
    InvariantAwareObject,
    PositiveIntegerInvariant,
)

from .commandinvariant import (
    InvariantAwareCommand,
)

#===============================================================================
# Named Tuples
#===============================================================================

#===============================================================================
# Helpers
#===============================================================================

def parse_image_base(line):
    """
    >>> l='        140000000 image base (0000000140000000 to 00000001409DAFFF)'
    >>> parse_image_base(l)
    ('14', '0000000140000000', '00000001409DAFFF')
    """

    ls = line.lstrip()
    base = ls[:ls.find(' ')].replace('0', '')

    ix1 = ls.find('(')+1
    ix2 = ls.find(' ', ix1+1)
    start = ls[ix1:ix2]

    ix1 = ls.find('0', ix2)
    ix2 = ls.find(')', ix1)
    end = ls[ix1:ix2]

    return (base, start, end)

def save_array_plot_to_png_file(filename, a):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Turn off interactive mode to disable plots being displayed
    # prior to saving them to disk.
    plt.ioff()


    plt.plot(a)
    plt.savefig(filename)


#===============================================================================
# Exceptions
#===============================================================================
class NotCfInstrumentedError(BaseException):
    pass

#===============================================================================
# Classes
#===============================================================================

class Dumpbin(InvariantAwareObject):

    path = None
    _path = None
    class PathArg(PathInvariant):
        pass

    def __init__(self, path):
        InvariantAwareObject.__init__(self)
        self.path = path
        self.conf = get_or_create_config()
        self._image_base_line = None
        self._image_base_lineno = None
        self._guard_cf_func_table_lineno = None
        self._guard_cf_targets_start = None
        self._guard_cf_targets_end = None
        self._section_header_4_lineno = None

        self._load()

    def _load(self):

        cmd = [
            self.conf.dumpbin_exe_path,
            '/headers',
            '/loadconfig',
            self.path
        ]

        from subprocess import check_output
        text = check_output(cmd).decode(sys.stdout.encoding)

        # Add a dummy line at the start of the array so that we can index
        # lines directly by line number instead of having to subtract one
        # first (to account for 0-based indexing).

        lines = [ '', ] + text.splitlines()

        guard_cf_targets_start = -1
        accumulating_cf_targets = False

        for (i, line) in enumerate(lines):
            if 'image base' in line:
                self._image_base_lineno = i
                self._image_base_line = line
                parsed = parse_image_base(line)
                self._image_base_chars = parsed[0]
                self._image_base_start = parsed[1]
                self._image_base_end = parsed[2]
            elif line.startswith('    Guard CF Function Table'):
                self._guard_cf_func_table_lineno = i
                guard_cf_targets_start = i + 4
            elif i == guard_cf_targets_start:
                accumulating_cf_targets = True
            elif accumulating_cf_targets and not line:
                self._guard_cf_targets_start = guard_cf_targets_start
                self._guard_cf_targets_end = i
                accumulating_cf_targets = False
            elif line.startswith('SECTION HEADER #4'):
                self._section_header_4_lineno = i

        self.text = text
        self.lines = lines

    @property
    def is_cf_instrumented(self):
        return self._guard_cf_func_table_lineno is not None

    @property
    @memoize
    def guard_cf_func_table_lines(self):
        if not self.is_cf_instrumented:
            raise NotCfInstrumentedError()

        start = self._guard_cf_targets_start
        end = self._guard_cf_targets_end
        assert end is not None and end > start, (start, end)
        return self.lines[start:end]

    @property
    def image_base_start(self):
        return self._image_base_start

    @property
    def image_base_end(self):
        return self._image_base_end

    @property
    def image_base_chars(self):
        return self._image_base_chars

    @property
    @memoize
    def guard_cf_func_table_addresses_base0(self):
        chars = self._image_base_chars
        zeros = '0' * len(chars)

        lines = []

        for line in self.guard_cf_func_table_lines:
            l = line[10:26].replace(chars, zeros, 1)[-10:]
            lines.append(l)

        return lines

    def save(self, output_dir):

        from .path import (
            basename,
            splitext,
            join_path,
        )

        filename = basename(self.path)
        name = splitext(filename)[0]

        addresses = self.guard_cf_func_table_addresses_base0
        values = [ int(address, base=16) for address in addresses ]

        prefix = join_path(output_dir, '%s-%d.' % (name, len(values)))

        text_path = ''.join((prefix, '.txt'))
        plot_path = ''.join((prefix, '.png'))
        binary_path = ''.join((prefix, '.keys'))

        with open(text_path, 'w') as f:
            f.write('\n'.join(addresses))

        import numpy as np

        a = np.array(values)

        save_array_plot_to_png_file(plot_path, a)

        fp = np.memmap(binary_path, dtype='uint32', mode='w+', shape=a.shape)
        fp[:] = a[:]
        del fp

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
