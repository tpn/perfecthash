#===============================================================================
# Imports
#===============================================================================

import re
import sys
import os.path

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
    (5368709120, '0000000140000000', '00000001409DAFFF')
    >>> l='       160880000 image base (0000000160880000 to 0000000160B26FFF)'
    >>> parse_image_base(l)
    (5914492928, '0000000160880000', '0000000160B26FFF')
    """

    ls = line.lstrip()
    base = int(ls[:ls.find(' ')], base=16)

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

    def __init__(self, path, disasm=None):
        InvariantAwareObject.__init__(self)
        self.path = path
        self.conf = get_or_create_config()
        self._image_base_line = None
        self._image_base_lineno = None
        self._guard_cf_func_table_lineno = None
        self._guard_cf_targets_start = None
        self._guard_cf_targets_end = None
        self._section_header_4_lineno = None
        self._save_plot = False

        self._load()

        if disasm:
            self._load_disasm()

    def _load(self):

        if os.path.exists(self.conf.dumpbin_exe_path):
            dumpbin_exe = self.conf.dumpbin_exe_path
        else:
            dumpbin_exe = 'dumpbin.exe'

        cmd = [
            dumpbin_exe,
            '/headers',
            '/loadconfig',
            self.path
        ]

        from subprocess import check_output
        raw = check_output(cmd)
        text = raw.decode(sys.stdout.encoding)

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
                self._image_base = parsed[0]
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

    def _load_disasm(self):

        import numpy as np

        if os.path.exists(self.conf.dumpbin_exe_path):
            dumpbin_exe = self.conf.dumpbin_exe_path
        else:
            dumpbin_exe = 'dumpbin.exe'

        cmd = [
            dumpbin_exe,
            '/disasm',
            self.path
        ]

        from subprocess import check_output
        raw = check_output(cmd)
        text = raw.decode(sys.stdout.encoding)

        # Add a dummy line at the start of the array so that we can index
        # lines directly by line number instead of having to subtract one
        # first (to account for 0-based indexing).

        lines = [ '', ] + text.splitlines()

        self.disasm_text = text
        self.disasm_lines = lines

        (func_linenos, names) = zip(*[
            (i, l[:-1]) for (i, l) in enumerate(lines) if (
                l and l[0] != ' ' and l[-1] == ':'
            )
        ])
        self.disasm_func_linenos = np.array(func_linenos)
        self.disasm_func_names = names

        if '_penter' in text:
            base = self.image_base
            call_ilt = 'call        @ILT'
            call_penter = 'call        _penter'
            (linenos, addresses) = zip(*[
                (i, ((int(l[2:18], base=16)) - base))
                    for (i, l) in enumerate(lines) if (
                        l and (
                            l.endswith('(_penter)') and call_ilt in l
                        ) or l.endswith(call_penter)
                    )
            ])

            self.disasm_call_penter_linenos = np.array(linenos)
            self.disasm_call_penter_rips = np.array(addresses, dtype='uint32')

            rip_to_name = { }
            func_linenos = self.disasm_func_linenos
            for (lineno, rip) in zip(linenos, addresses):
                i = np.searchsorted(func_linenos, v=lineno) - 1
                name = names[i]
                rip_to_name[rip] = name

            self.penter_rip_to_name = rip_to_name
            self.penter_names = np.array(list(rip_to_name.values()), dtype='str')

        if '__Pogo' in text:
            base = self.image_base
            call_prefix = 'call        '
            (linenos, addresses) = zip(*[
                (i, ((int(l[2:18], base=16)) - base))
                    for (i, l) in enumerate(lines) if (
                        '__Pogo' in l and call_prefix in l
                    )
            ])

            self.disasm_call_pogo_linenos = np.array(linenos)
            self.disasm_call_pogo_rips = np.array(addresses, dtype='uint32')

            pogo_rip_to_name = { }
            func_linenos = self.disasm_func_linenos
            for (lineno, rip) in zip(linenos, addresses):
                i = np.searchsorted(func_linenos, v=lineno) - 1
                name = names[i]
                pogo_rip_to_name[rip] = name

            self.pogo_rip_to_name = pogo_rip_to_name

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
        return int(self._image_base_start, base=16)

    @property
    def image_base_end(self):
        return int(self._image_base_end, base=16)

    @property
    def image_base(self):
        return self._image_base

    @property
    def image_size(self):
        return self.image_base_end - self.image_base_start

    @property
    @memoize
    def guard_cf_func_table_addresses(self):
        return [ l[10:26][-10:] for l in self.guard_cf_func_table_lines ]

    @property
    @memoize
    def guard_cf_func_table_address_values(self):
        addresses = self.guard_cf_func_table_addresses
        values = [ int(address, base=16) for address in addresses ]
        return values

    @property
    @memoize
    def guard_cf_func_table_address_array_base0(self):
        values = self.guard_cf_func_table_address_values
        import numpy as np
        b = np.array(values)
        a = b - self.image_base
        return a

    @property
    @memoize
    def guard_cf_func_table_addresses_base0(self):
        array = self.guard_cf_func_table_address_array_base0
        return [ hex(i)[2:].zfill(8).encode('ascii') for i in array ]

    def binary_path(self, output_dir, a):
        from .path import (
            basename,
            splitext,
            join_path,
        )

        filename = basename(self.path)
        name = splitext(filename)[0]

        first_char = name[0]
        if first_char >= '0' and first_char <= '9':
            name = f'_{name}'

        name = name.replace(' ', '').replace('(', '').replace(')', '')

        prefix = join_path(output_dir, '%s-%d.' % (name, len(a)))

        binary_path = ''.join((prefix, '.keys'))

        return binary_path

    def plot_path(self, output_dir):
        return self.binary_path(output_dir).replace('.keys', '.png')

    def _save_rips_as_keys(self, rips, suffix):

        import numpy as np

        from .path import (
            dirname,
            join_path,
        )

        a = np.sort(np.unique(rips))

        binary_path = self.path.replace('.dll', f'{suffix}.keys')
        fp = np.memmap(binary_path, dtype='uint32', mode='w+', shape=a.shape)
        fp[:] = a[:]
        del fp

        print(f'Wrote {binary_path}.')

    def save_penter_rips(self):
        self._save_rips_as_keys(self.disasm_call_penter_rips, 'PenterRips')

    def save_pogo_rips(self):
        self._save_rips_as_keys(self.disasm_call_pogo_rips, 'PogoRips')

    def save(self, output_dir):

        import numpy as np

        a = self.guard_cf_func_table_address_array_base0
        a = np.sort(np.unique(np.array(a, dtype='uint32')))
        # Skip keys of just 0.
        if a[0] == 0:
            a = a[1:]

        if len(a) == 0:
            return

        binary_path = self.binary_path(output_dir)

        if self._save_plot:
            plot_path = self.plot_path(output_dir)
            save_array_plot_to_png_file(plot_path, a)

        fp = np.memmap(binary_path, dtype='uint32', mode='w+', shape=a.shape)
        fp[:] = a[:]
        del fp

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
