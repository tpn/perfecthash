#===============================================================================
# Imports
#===============================================================================
import os
import re
import sys
import stat
import base64
import inspect
import subprocess

from abc import (
    abstractproperty,
)

from os.path import (
    isdir,
    isfile,
    abspath,
    dirname,
    basename,
    expanduser,
    expandvars,
)

from textwrap import dedent

try:
    from ConfigParser import (
        NoOptionError,
        NoSectionError,
        RawConfigParser,
    )
except ImportError:
    from configparser import (
        NoOptionError,
        NoSectionError,
        RawConfigParser,
    )

from .path import (
    join_path,
)

from .util import (
    memoize,
    iterable,
    classproperty,

    Dict,
    Options,
)

#===============================================================================
# Globals
#===============================================================================

CONFIG = None
CONFIG_CLASS = None

PROGRAM_NAME = None
COMMAND_NAME = None
COMMAND_MODULES = None

PATH = dirname(abspath(__file__))

NAMESPACE = basename(PATH)

LIB_DIR  = join_path(PATH, '..')
SRC_DIR  = join_path(LIB_DIR, '../src')
BIN_DIR  = join_path(LIB_DIR, '../bin')
CONF_DIR = join_path(LIB_DIR, '../conf')
LOGS_DIR = join_path(LIB_DIR, '../logs')
ROOT_DIR = join_path(LIB_DIR, '../../')
DEBUG_DIR = join_path(ROOT_DIR, 'x64/Debug')
RELEASE_DIR = join_path(ROOT_DIR, 'x64/Release')

#fixme: revisit these assertions
#assert LIB_DIR.endswith('lib'), LIB_DIR
#
#for d in (LIB_DIR, BIN_DIR, CONF_DIR, LOGS_DIR):
#    assert isdir(d), d

# HOSTFQDN may have the FQDN or it may not; HOSTNAME will always be the
# shortest representation of the hostname.
HOSTFQDN = subprocess.check_output('hostname')[:-len(os.linesep)].lower()
try:
    HOSTNAME = HOSTFQDN.split('.')[0]
except TypeError:
    HOSTFQDN = HOSTFQDN.decode('utf-8')
    HOSTNAME = HOSTFQDN.split('.')[0]

#===============================================================================
# Exceptions
#===============================================================================
class ConfigError(BaseException):
    pass

class NoConfigObjectCreated(BaseException):
    pass

class ConfigObjectAlreadyCreated(BaseException):
    pass

class ConfigClassAlreadySet(BaseException):
    pass

#===============================================================================
# Helpers
#===============================================================================
def get_config():
    global CONFIG
    if not CONFIG:
        raise NoConfigObjectCreated()
    return CONFIG

def get_or_create_config():
    try:
        conf = get_config()
    except NoConfigObjectCreated:
        conf = Config()
        conf.load()
    return conf

def _clear_config_if_already_created():
    global CONFIG
    if CONFIG:
        CONFIG = None

def set_config_class(cls):
    assert cls
    global CONFIG_CLASS
    if CONFIG_CLASS:
        raise ConfigClassAlreadySet()
    CONFIG_CLASS = cls

def _clear_config_class_if_already_set():
    global CONFIG_CLASS
    if CONFIG_CLASS:
        CONFIG_CLASS = None

#===============================================================================
# Classes
#===============================================================================
class Config(RawConfigParser):

    def __init__(self, options=None):
        RawConfigParser.__init__(self)

        self.optionxform = str

        self.options = options if options else Options()
        self.hostname = HOSTFQDN
        self.shortname = HOSTNAME

        self.files = None
        self.filename = None
        self._is_production = None
        self._multiline_pattern = re.compile(r'([^\s].*?)([\s]+\\)?')

        global CONFIG
        if CONFIG is not None:
            raise ConfigObjectAlreadyCreated()
        CONFIG = self

    def get_multiline_csv_as_list(self, section, name):
        return self._csv_as_list(
            self.get_multiline_to_single_line(section, name)
        )

    def get_csv_as_list(self, section, name):
        return self._csv_as_list(self.get(section, name))
        return [ n.strip() for n in csv.split(',') ]

    def _csv_as_list(self, csv):
        return [ n.strip() for n in csv.split(',') ]

    def get_multiline_to_single_line(self, section, name):
        return (
            self.get(section, name)
                .replace(os.linesep, '')
                .replace('\\', '')
                .strip()
        )

    def get_multiline(self, section, name):
        # I'm sure there's a more efficient way of doing this.
        value = self.get(section, name)
        if not value:
            return

        output = list()
        lines = value.split(os.linesep)
        pattern = re.compile(r'(.+?)([\s]*\\)')
        for line in lines:
            if line.endswith('\\'):
                matches = pattern.findall(line)
                if matches:
                    output.append(matches[0][0])
            elif line:
                output.append(line)

        joined = '\n'.join(output)
        return joined


    @classproperty
    @classmethod
    def namespace(cls):
        return basename(dirname(inspect.getsourcefile(cls)))

    @classmethod
    def _resolve_dir(cls, name=None):
        path = inspect.getsourcefile(cls)
        base = dirname(join_path(path, '../..'))
        return join_path(base, name) if name else base

    @classproperty
    @classmethod
    def project_root(cls):
        return cls._resolve_dir()

    @classproperty
    @classmethod
    def lib_dir(cls):
        return cls._resolve_dir('lib')

    @classproperty
    @classmethod
    def bin_dir(cls):
        return cls._resolve_dir('bin')

    @classproperty
    @classmethod
    def conf_dir(cls):
        return cls._resolve_dir('conf')

    @classproperty
    @classmethod
    def logs_dir(cls):
        return cls._resolve_dir('logs')

    @classproperty
    @classmethod
    def data_dir(cls):
        d = cls._resolve_dir('data')
        if not isdir(d):
            os.makedirs(d)
        return d

    @classproperty
    @classmethod
    def parent(cls):
        try:
            return cls.__base__
        except AttributeError:
            return None

    def _absdir(self, name, section='main'):
        count = 0
        total = 0
        max_total = 10
        p = self.get(section, name)
        if not p:
            return
        while count != 2 and total < max_total:
            count = 0
            total += 1
            if '~' in p:
                p = expanduser(p)
                count -= 1
            else:
                count += 1

            if '%' in p or '$' in p:
                p = expandvars(p)
                count -= 1
            else:
                count += 1

        if total == max_total:
            args = (section, name)
            msg = "Exceeded user/var path recursion depth for %s.%s." % args
            raise RuntimeError(msg)

        return abspath(p)

    @property
    @memoize
    def src_dir(self):
        return self._absdir('src_dir')

    @classmethod
    def discover_config_files(cls, files=None):
        if files is None:
            assert cls.parent
            files = list()

        files.append((cls.conf_dir, cls.namespace))

        try:
            if cls.parent:
                cls.parent.discover_config_files(files)
        except AttributeError:
            pass

    def post_load(self):
        """
        Called after load() has run.  Implement in sublcass to prime
        additional attributes/settings once config is available.
        """
        pass

    def load(self, filename=None):
        info = []
        self.discover_config_files(info)

        files = []
        for (conf_dir, namespace) in info:
            prefix = join_path(conf_dir, '%s%s.conf' % (namespace, '%s'))

            short = (self.hostname != self.shortname)
            upper = namespace.upper()

            files += [
                f for f in [
                    prefix % '',
                    prefix % ('-' + self.hostname),
                    prefix % ('-' + self.shortname) if short else None,
                    os.environ.get('%s_CONF' % upper) or None,
                ] if f
            ]

        if filename:
            files.append(filename)

        with open(files[0], 'r') as f:
            self.readfp(f, files[0])

        self.read(files[1:])

        self.files = files
        self.filename = filename

        self.post_load()

    @property
    @memoize
    def perfecthash_x64_dll_debug_path(self):
        return join_path(
            self.project_root,
            self.get('perfecthash', 'perfecthash_x64_dll_debug'),
        )

    @property
    @memoize
    def perfecthash_x64_dll_release_path(self):
        return join_path(
            self.project_root,
            self.get('perfecthash', 'perfecthash_x64_dll_release'),
        )

    @property
    @memoize
    def current_architecture(self):
        import ctypes
        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        return 'x64' if ptr_size == 8 else 'x86'

    @property
    @memoize
    def perfecthash_dll_path(self):
        current_dll_type = self.get('perfecthash', 'current_dll_type')
        current_arch = self.current_architecture
        fmt = 'perfecthash_%s_dll_%s_path'
        prop = fmt % (current_arch, current_dll_type)
        return getattr(self, prop)

    @property
    @memoize
    def link_exe_path(self):
        return self.get('perfecthash', 'link_exe')

    @property
    @memoize
    def dumpbin_exe_path(self):
        return self.get('perfecthash', 'dumpbin_exe')

    @property
    @memoize
    def base_output_dir(self):
        return self._absdir('base_output_dir', section='perfecthash')

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
