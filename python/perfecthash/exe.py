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
ph = InProcessPythonCommandLineProcessWrapper('ph', 'perfecthash')

# vim:set ts=8 sw=4 sts=4 tw=78 et:
