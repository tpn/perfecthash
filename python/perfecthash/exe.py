#===============================================================================
# Imports
#===============================================================================

from functools import partial

from .util import (
    ProcessWrapper,
)

#===============================================================================
# Classes
#===============================================================================
class SubversionClientException(Exception):
    pass

class SubversionClient(ProcessWrapper):
    # Purpose-built extension, intended for helping with testing.
    def __init__(self, exe, *args, **kwds):
        ProcessWrapper.__init__(self, exe, *args, **kwds)
        self.username = str()
        self.password = str()
        self.exception_class = SubversionClientException

    def build_command_line(self, exe, action, *args, **kwds):
        #import wingdbstub
        args = list(args)
        kwds = dict(kwds)

        is_ra = False
        if action == 'ci':
            is_ra = True
        elif action in ('cp', 'copy', 'mv', 'move', 'mkdir', 'rm', 'remove'):
            line = ' '.join(args)
            is_ra = (
                line.count('file://') or
                line.count('svn://')  or
                line.count('http://')
            )

        if is_ra:
            assert self.username and self.password
            kwds['username'] = self.username
            kwds['password'] = self.password
            kwds['no_auth_cache']   = True
            kwds['non_interactive'] = True

            if 'm' not in kwds:
                kwds['m'] = '""'

            if 'm' in kwds:
                m = kwds['m'] or '""'
                if not m.startswith('"'):
                    m = '"' + m
                if not m.endswith('"'):
                    m = m + '"'
                kwds['m'] = m

            if 'u' in kwds:
                kwds['username'] = kwds['u']
                del kwds['u']

        return ProcessWrapper.build_command_line(self, exe, action,
                                                 *args, **kwds)

class InProcessPythonCommandLineProcessWrapper(ProcessWrapper):
    """
    Wraps our .cli machinery instead of calling out to a subprocess.
    """
    def __init__(self, program_name, module_name):
        import cli
        cli.INTERACTIVE = True
        self.program_name = program_name
        self.module_name = module_name
        self.main = partial(cli.main, program_name, module_name)

        # Explicitly prime all the subcommands up-front as attributes so that
        # tab-completion works properly within IPython.  (Future enhancement:
        # return something more sophisticated than a partial to self.execute;
        # i.e. something that leverages all the command docstrings etc.)
        self._subcommands = None
        self._load_subcommands()
        for name in self._subcommands:
            func = partial(self.execute, name)
            setattr(self, name, func)

    def _init_command_list(self, exe, action, *args, **kwds):
        return [ action.replace('_', '-') ]

    def execute(self, action, *args, **kwds):
        cmd = self.build_command_line(self, action, *args, **kwds)

        self.cli = self.main(cmd)

        if self.cli.returncode:
            return

        return self.cli.commandline.command

    def _load_subcommands(self):
        from .command import silence_streams
        with silence_streams():
            cli = self.main(argv=[])

        self._subcommands = [
            c.replace('-', '_')
                for c in cli._commands_by_name
        ]

#===============================================================================
# Instances
#===============================================================================
t = InProcessPythonCommandLineProcessWrapper('t', 'perfecthash')

svn = SubversionClient('svn')
svnmucc = SubversionClient('svnmucc')
svnadmin = ProcessWrapper('svnadmin')
evnadmin = ProcessWrapper('evnadmin')

# vim:set ts=8 sw=4 sts=4 tw=78 et:
