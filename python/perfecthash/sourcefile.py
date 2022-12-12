#===============================================================================
# Imports
#===============================================================================
import re
import os.path
import sys
import textwrap

from collections import namedtuple

from .path import (
    abspath,
    dirname,
    join_path,
)

from .util import (
    memoize,
    align_trailing_slashes,
    strip_linesep_if_present,
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
Function = namedtuple(
    'Function',
    ['lineno', 'length', 'typedef', 'funcname']
)

FunctionDefinition = namedtuple(
    'FunctionDefinition', [
        'funcname',
        'first_line',
        'last_line',
        'first_block_line',
        'last_block_line',
        'last_return_line',
    ]
)

MultilineMacroDefinition = namedtuple(
    'MultilineMacroDefinition', [
        'name',
        'first_lineno',
        'last_lineno',
        'lines',
    ]
)

MultilineConstStringDecl = namedtuple(
    'MultilineConstStringDecl', [
        'name',
        'first_lineno',
        'last_lineno',
        'lines',
    ]
)

#===============================================================================
# Globals
#===============================================================================
THIS_DIR = dirname(abspath(__file__))
PERFECT_HASH_PDBEX_HEADER_PATH = (
    join_path(
        THIS_DIR,
        '../../src/x64/Release/PerfectHashPdbexHeader.h'
    )
)

PERFECT_HASH_DLL_PATH = (
    join_path(
        THIS_DIR,
        '../../src/x64/Release/PerfectHash.dll'
    )
)

PERFECT_HASH_PDB_PATH = PERFECT_HASH_DLL_PATH.replace('.dll', '.pdb')

PDBEX_EXE_PATH = (
    join_path(
        THIS_DIR,
        '../../bin/pdbex.exe'
    )
)

#===============================================================================
# Helpers
#===============================================================================
def convert_function_decls_to_funcptr_typedefs(text, prefix=None, prepend=None):
    if not prefix:
        prefix = '^BOOL\nIMAGEAPI\n'
    if not prepend:
        prepend = 'typedef\n_Check_return_\n_Success_(return != 0)'.split('\n')
    pre_lines = prefix.count('\n')
    name_line = pre_lines
    pattern = '%s.*?    \);$' % prefix
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    matches = regex.findall(text)
    regex = re.compile('[A-Z][^A-Z]*')
    results = []
    for match in matches:
        match_lines = match.splitlines()
        name = match_lines[name_line][:-1]
        tokens = regex.findall(name)
        upper_tokens = [ t.upper() for t in tokens ]
        cap_name = '_'.join(upper_tokens)
        func_lines = (
            prepend +
            match_lines[0:pre_lines] +
            [ '(%s)(' % cap_name ] +
            match_lines[name_line+1:] +
            [ 'typedef %s *P%s;\n' % (cap_name, cap_name) ]
        )
        func_text = '\n'.join(func_lines)
        results.append(func_text)

    return results

def generate_sqlite3_column_func_switch_statement(tablename, mdecl):
    lines = [
        '',
        '        //',
        '        // Begin auto-generated section.',
        '        //',
    ]
    bigint_casts = (
        'LARGE_INTEGER',
        'FILETIME',
        'SYSTEMTIME',
    )
    # Skip the first two lines and last line.
    for (i, line) in enumerate(mdecl.lines[2:-1]):

        # Find the index of the first quote.
        ix = line.find('"')
        assert ix != -1, line

        # Find the index of the second quote.
        ix2 = line.find('"', ix+1)
        assert ix2 != -1, (line, ix1)

        # Extract the schema part and omit the trailing ", " if applicable.
        schema = line[ix+1:ix2]
        if schema.endswith(', '):
            schema = schema[:-2]
        (name, dtype) = schema.split(' ')
        if '_' in name:
            field = name.replace('_', '->')
        else:
            field = name

        # Quick hack for MetadataInfo.  We only need one ->.
        if field.startswith('MetadataInfo->'):
            fields = field.split('->')
            field = '%s.%s' % ('->'.join(fields[:2]), '.'.join(fields[2:]))

        predicate = None

        # Find the start of the line's comment.
        ix3 = line.find('//', ix2 + 1)
        if ix3 == -1:
            # No access descriptor for this field.  Make the target the name
            # of the table (e.g. 'Address') plus the name of the field, e.g.
            # Address->BaseAddress.
            if '_' not in name:
                access = '%s->%s' % (tablename, name)
            else:
                access = field
        else:
            # Extract the access descriptor.  If the field is a BIGINT, check
            # to see if the access descriptor ends with LARGE_INTEGER and that
            # there is no comma -- this is an indicator that we can use the
            # tablename + mdecl.name approach above, but should use the cast
            # for RESULT_[PU]LARGE_INTEGER.
            access = line[ix3+3:]
            # Check to see if there's a predicate, indicated by the presence
            # of an opening square bracket and a subsequent closing square
            # bracket.
            ix4 = line.find('[', ix3+3)
            if ix4 != -1:
                ix5 = line.find(']', ix4 + 1)
                assert ix5 != -1, (line, ix4, ix5)
                predicate = line[ix4+1:ix5]
                access = access.replace(line[ix4:ix5+1], '')
                if access.endswith(', '):
                    access = access[:-2]

            if access.endswith(bigint_casts):
                if ',' not in access:
                    if '_' not in name:
                        access = '%s->%s, %s' % (tablename, name, access)
                    else:
                        access = '%s, %s' % (field, access)
            elif not access:
                if '_' not in name:
                    access = '%s->%s' % (tablename, name)
                else:
                    access = field

        stmt = None


        if dtype == 'TEXT':
            # TEXT should always have a "<type>" suffix, e.g.:
            #   Path->Full, UNICODE_STRING
            #   PSTRING
            # etc.
            if ',' in access:
                (target, cast) = access.split(', ')
            else:
                cast = access
                target = '%s->%s' % (tablename, name)
            stmt = 'RESULT_%s(%s);' % (cast, target)
        elif dtype.startswith('BLOB'):
            # BLOB should be followed by either ", sizeof()" or "sizeof(*)".
            (target, cast) = access.split(', ')
            assert cast in ('sizeof()', 'sizeof(*)'), (cast, line, dtype)
            if '*' in cast:
                ptr = 'P'
                sizeof = 'sizeof(*%s)' % target
            else:
                ptr = ''
                sizeof = 'sizeof(%s)' % target
            fmt = 'RESULT_%s%s(%s, %s);'
            stmt = fmt % (ptr, dtype, target, sizeof)
        elif dtype in ('REAL', 'DOUBLE', 'FLOAT'):
            if ', ' in access:
                (target, cast) = access.split(', ')
            else:
                target = access
                cast = 'DOUBLE'
            fmt = 'RESULT_%s((%s)%s);'
            stmt = fmt % (cast, cast, target)
        elif dtype == 'NUMERIC':
            raise RuntimeError("NUMERIC not supported.")
        else:
            assert 'INT' in dtype, (dtype, line)
            is_big = 'BIG' in dtype
            if ', ' in access:
                (target, cast) = access.split(', ')
            else:
                target = access
                cast = 'ULONGLONG' if is_big else 'ULONG'
            fmt = 'RESULT_%s(%s);'
            stmt = fmt % (cast, target)

        lines += [
            '',
            '        //',
            '        // %d: %s %s' % (i, name, dtype),
            '        //',
            '',
            '        case %d:' % i,
        ]

        if not predicate:
            lines += [
                '            %s' % stmt,
                '            break;',
            ]
        else:
            lines += [
                '            if (!(%s)) {' % predicate,
                '                RESULT_NULL();',
                '            } else {',
                '                %s' % stmt,
                '            }',
                '            break;',
            ]

    lines += [
        '',
        '        default:',
        '           INVALID_COLUMN();',
        '',
        '        //',
        '        // End auto-generated section.',
        '        //',
        '',
    ]

    return lines

#===============================================================================
# Classes
#===============================================================================
class SourceFile(InvariantAwareObject):
    path = None
    _path = None
    class PathArg(PathInvariant):
        pass

    def __init__(self, path):
        InvariantAwareObject.__init__(self)
        self.path = path

    @property
    @memoize
    def data(self):
        with open(self._path, 'r') as f:
            return f.read()

    @property
    @memoize
    def lines(self):
        return self.data.splitlines()

    @memoize
    def lines_as_cstr(self):
        def process(line):
            line = line.replace('\\', '\\\\')
            line = line.replace('"', '\\"')
            line = '"%s\\n"' % line
            return line

        return [ process(line) for line in self.lines ]

    @property
    @memoize
    def defines(self):
        results = []
        for (lineno, line) in enumerate(self.lines):
            if line.startswith('#define'):
                results.append((lineno, line))
        return results

    @property
    @memoize
    def lines_ending_with_backslash(self):
        results = []
        for (lineno, line) in enumerate(self.lines):
            if line.endswith('\\'):
                results.append((lineno, line))
        return results

    @property
    @memoize
    def blank_lines(self):
        results = []
        for (lineno, line) in enumerate(self.lines):
            if not line or not line.replace(' ', ''):
                results.append((lineno, line))
        return results

    @property
    @memoize
    def multiline_macro_defines(self):
        results = {}

        for define in self.defines:
            lines = []
            (lineno, line) = define
            if not line.endswith('\\'):
                continue

            name = line.replace('#define ', '')
            ix = name.find('(')
            if ix == -1:
                ix = name.find(' ')
            name = name[:ix]

            first_lineno = lineno
            lines.append(line)

            lineno += 1
            line = self.lines[lineno]

            num_lines = 0
            while line.endswith('\\'):
                lines.append(line)
                lineno += 1
                line = self.lines[lineno]
                num_lines += 1

            if num_lines == 0:
                continue

            last_lineno = lineno
            lines.append(line)

            results[name] = MultilineMacroDefinition(
                name=name,
                first_lineno=first_lineno,
                last_lineno=last_lineno,
                lines=lines
            )

        return results

    @property
    @memoize
    def const_string_decls(self):
        results = []
        string_types = ('CHAR', 'WCHAR', 'STR', 'WSTR')
        for (lineno, line) in enumerate(self.lines):
            if not line.startswith('CONST '):
                continue

            tokens = line.split(' ')
            if not any(token in string_types for token in tokens):
                continue

            if line.endswith('[] ='):
                results.append((lineno, line))

        return results

    @property
    @memoize
    def const_array_decls(self):
        results = []
        for (lineno, line) in enumerate(self.lines):
            if not line.startswith('CONST '):
                continue

            if line.endswith('[] = {'):
                results.append((lineno, line))

        return results

    @property
    @memoize
    def multiline_const_string_decls(self):
        results = {}

        decls = self.const_string_decls
        for decl in decls:
            lines = []
            (lineno, line) = decl

            name = line.replace('CONST ', '').split(' ')[1].replace('[]', '')
            first_lineno = lineno
            lines.append(line)

            lineno += 1
            line = self.lines[lineno]

            while not line.endswith(';'):
                lines.append(line)
                lineno += 1
                line = self.lines[lineno]

            last_lineno = lineno
            lines.append(line)

            results[name] = MultilineConstStringDecl(
                name=name,
                first_lineno=first_lineno,
                last_lineno=last_lineno,
                lines=lines
            )

        return results

    def function_definition(self, funcname, block=None):
        partial = False
        found = False
        func_line = None
        for (lineno, line) in enumerate(self.lines):
            if not partial:
                if line.startswith(funcname):
                    partial = True
                    func_line = line
                    continue
            else:
                if line == '{':
                    found = True
                    break
                elif line.startswith('    '):
                    continue
                else:
                    partial = False
                    continue

        if not found:
            return None

        i = lineno-1
        prev_line = self.lines[i]
        while prev_line:
            i -= 1
            prev_line = self.lines[i]
        first_line = i

        first_block = None
        last_block = None
        last_return = None

        block_length = len(block) if block else None

        i = lineno+2
        next_line = self.lines[i]
        while next_line != '}':
            if next_line == '    }':
                last_block = i
            elif next_line.startswith('    return '):
                last_return = i

            if block:
                next_block = self.lines[i:i+block_length]
                if block == next_block:
                    if not first_block:
                        first_block = i

            i += 1
            next_line = self.lines[i]

        last_line = i

        return FunctionDefinition(
            funcname,
            first_line,
            last_line,
            first_block,
            last_block,
            last_return,
        )

    @memoize
    def functions_from_multiline_define(self, name):
        results = []
        macro = self.multiline_macro_defines[name]
        for (lineno, line) in enumerate(macro.lines):
            if line.startswith('#define'):
                continue
            length = len(line)
            line = line[4:line.find(';')]
            (typedef, funcname) = line.split()
            results.append(Function(lineno, length, typedef, funcname))
        return results


class HeaderFile(SourceFile):
    pass

class CodeFile(SourceFile):
    pass

class PerfectHashPdbexHeaderFile(SourceFile):
    path = None
    _path = None
    class PathArg(PathInvariant):
        pass

    def __init__(self, path=None):
        InvariantAwareObject.__init__(self)
        if not path:
            path = PERFECT_HASH_PDBEX_HEADER_PATH

        run_pdbex = False

        try:
            self.path = path
        except:
            run_pdbex = True
        else:

            # Check if the header is newer than the .dll file.
            st_header = os.stat(self.path)
            st_dll = os.stat(PERFECT_HASH_DLL_PATH)

            if st_header.st_mtime < st_dll.st_mtime:
                run_pdbex = True

        if run_pdbex:
            cmd = [
                PDBEX_EXE_PATH,
                '*',                          # Symbols
                f'{PERFECT_HASH_PDB_PATH}',   # PDB file
                '-o',                           # Output file
                f'{PERFECT_HASH_PDBEX_HEADER_PATH}', # PdbexHeader file
                '-p',   # Create padding members
                '-m',   # Create Microsoft typedefs
                '-d',   # Allow unnamed data types
                '-i',   # Use types from stdint.h instead of native types
                '-j',   # Print definitions of referenced types
                '-k',   # Print header
                '-n',   # Print declarations
                '-l',   # Print definitions
                '-f',   # Print functions
                '-z',   # Print #pragma pack directives
            ]

            print('Running pdbex...')
            from subprocess import check_call
            check_call(cmd)
            if not self._path:
                self.path = path

    def find_lines_startingwith(self, string):
        results = []
        for (lineno, line) in enumerate(self.lines):
            if line.startswith(string):
                results.append((lineno, line))
        return results

    def find_closing_brace_line_numbers(self):
        results = []
        for (lineno, line) in enumerate(self.lines):
            if line.startswith('}'):
                results.append(lineno)
        return results

    @property
    @memoize
    def closing_brace_line_numbers_as_array(self):
        import numpy as np
        return np.array(self.find_closing_brace_line_numbers())

    def extract_elements(self, elements):
        results = {}
        lines = self.lines
        linenos = self.closing_brace_line_numbers_as_array
        for (start_lineno, line) in elements:
            end_lineno = linenos[linenos.searchsorted(start_lineno+1)]
            elem_lines = lines[start_lineno:end_lineno+1]
            last_line = elem_lines[-1]
            name = last_line.split(',')[0][2:]
            results[name] = elem_lines
        return results

    @property
    @memoize
    def structs(self):
        elems = self.find_lines_startingwith('typedef struct _PERFECT_HASH')
        return self.extract_elements(elems)

    @property
    @memoize
    def enums(self):
        elems = self.find_lines_startingwith('typedef enum _PERFECT_HASH')
        return self.extract_elements(elems)

    def parse_enums(self, name, skip_if_startswith=None,
                    remove=None, split_on=None):

        lines = self.enums[name][3:-2]

        results = []
        for line in lines:
            line = line[2:]
            if skip_if_startswith and line.startswith(skip_if_startswith):
                continue
            elif 'Null' in line or 'Invalid' in line:
                continue

            if remove:
                assert line.startswith(remove)
                line = line.replace(remove, '')

            if split_on:
                line = line.split(split_on)[0]

            results.append(line)
        return results

    @property
    @memoize
    def best_coverage_types(self):
        return self.parse_enums(
            'PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID',
            skip_if_startswith='PerfectHash',
            remove='BestCoverageType',
            split_on='Id ='
        )

    @property
    def disabled_hash_functions(self):
        funcs = self.parse_enums(
            'PERFECT_HASH_DISABLED_HASH_FUNCTION_ID',
            remove='PerfectHashDisabledHash',
            split_on='FunctionId ='
        )
        return set(funcs)

    @property
    def hash_functions(self):
        import ipdb
        ipdb.set_trace()
        exclude = self.disabled_hash_functions
        funcs = self.parse_enums(
            'PERFECT_HASH_HASH_FUNCTION_ID',
            remove='PerfectHashHash',
            split_on='FunctionId ='
        )
        return [ name for name in funcs if name not in exclude ]

    @property
    @memoize
    def table_create_parameters(self):
        return self.parse_enums(
            'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID',
            skip_if_startswith='PerfectHashTable',
            remove='TableCreateParameter',
            split_on='Id ='
        )

    @memoize
    def get_hash_function_code(self, algo, masking):
        results = {}
        for name in self.hash_functions:
            part = (
                '../../src/CompiledPerfectHashTable/'
                'CompiledPerfectHashTable'
                f'{algo}Index{name}{masking}.c'
            )
            path = join_path(THIS_DIR, part)
            with open(path, 'r') as f:
                data = f.read()
                ix = data.find('#ifndef CPH_INLINE_ROUTINES')
                assert ix != -1
                data = data[:ix-1]
                results[name] = data
        return results

    @property
    def graph_impls(self):
        # This isn't an enum yet.
        return ('1', '2', '3')

    @property
    def algorithms(self):
        return ('Chm01',)

    @property
    def maskings(self):
        return ('And',)

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
