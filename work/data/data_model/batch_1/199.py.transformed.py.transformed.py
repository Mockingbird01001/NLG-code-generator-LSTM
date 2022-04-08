import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
    Configuration,
    Kind,
    get_configuration_files,
    kinds,
)
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
logger = logging.getLogger(__name__)
class ConfigurationCommand(Command):
    ignore_require_venv = True
    usage =
    def add_options(self):
        self.cmd_opts.add_option(
            '--editor',
            dest='editor',
            action='store',
            default=None,
            help=(
                'Editor to use to edit the file. Uses VISUAL or EDITOR '
                'environment variables if not provided.'
            )
        )
        self.cmd_opts.add_option(
            '--global',
            dest='global_file',
            action='store_true',
            default=False,
            help='Use the system-wide configuration file only'
        )
        self.cmd_opts.add_option(
            '--user',
            dest='user_file',
            action='store_true',
            default=False,
            help='Use the user configuration file only'
        )
        self.cmd_opts.add_option(
            '--site',
            dest='site_file',
            action='store_true',
            default=False,
            help='Use the current environment configuration file only'
        )
        self.parser.insert_option_group(0, self.cmd_opts)
    def run(self, options, args):
        handlers = {
            "list": self.list_values,
            "edit": self.open_in_editor,
            "get": self.get_name,
            "set": self.set_name_value,
            "unset": self.unset_name,
            "debug": self.list_config_values,
        }
        if not args or args[0] not in handlers:
            logger.error(
                "Need an action (%s) to perform.",
                ", ".join(sorted(handlers)),
            )
            return ERROR
        action = args[0]
        try:
            load_only = self._determine_file(
                options, need_value=(action in ["get", "set", "unset", "edit"])
            )
        except PipError as e:
            logger.error(e.args[0])
            return ERROR
        self.configuration = Configuration(
            isolated=options.isolated_mode, load_only=load_only
        )
        self.configuration.load()
        try:
            handlers[action](options, args[1:])
        except PipError as e:
            logger.error(e.args[0])
            return ERROR
        return SUCCESS
    def _determine_file(self, options, need_value):
        file_options = [key for key, value in (
            (kinds.USER, options.user_file),
            (kinds.GLOBAL, options.global_file),
            (kinds.SITE, options.site_file),
        ) if value]
        if not file_options:
            if not need_value:
                return None
            elif any(
                os.path.exists(site_config_file)
                for site_config_file in get_configuration_files()[kinds.SITE]
            ):
                return kinds.SITE
            else:
                return kinds.USER
        elif len(file_options) == 1:
            return file_options[0]
        raise PipError(
            "Need exactly one file to operate upon "
            "(--user, --site, --global) to perform."
        )
    def list_values(self, options, args):
        self._get_n_args(args, "list", n=0)
        for key, value in sorted(self.configuration.items()):
            write_output("%s=%r", key, value)
    def get_name(self, options, args):
        key = self._get_n_args(args, "get [name]", n=1)
        value = self.configuration.get_value(key)
        write_output("%s", value)
    def set_name_value(self, options, args):
        key, value = self._get_n_args(args, "set [name] [value]", n=2)
        self.configuration.set_value(key, value)
        self._save_configuration()
    def unset_name(self, options, args):
        key = self._get_n_args(args, "unset [name]", n=1)
        self.configuration.unset_value(key)
        self._save_configuration()
    def list_config_values(self, options, args):
        self._get_n_args(args, "debug", n=0)
        self.print_env_var_values()
        for variant, files in sorted(self.configuration.iter_config_files()):
            write_output("%s:", variant)
            for fname in files:
                with indent_log():
                    file_exists = os.path.exists(fname)
                    write_output("%s, exists: %r",
                                 fname, file_exists)
                    if file_exists:
                        self.print_config_file_values(variant)
    def print_config_file_values(self, variant):
        for name, value in self.configuration.                get_values_in_config(variant).items():
            with indent_log():
                write_output("%s: %s", name, value)
    def print_env_var_values(self):
        write_output("%s:", 'env_var')
        with indent_log():
            for key, value in sorted(self.configuration.get_environ_vars()):
                env_var = f'PIP_{key.upper()}'
                write_output("%s=%r", env_var, value)
    def open_in_editor(self, options, args):
        editor = self._determine_editor(options)
        fname = self.configuration.get_file_to_edit()
        if fname is None:
            raise PipError("Could not determine appropriate file.")
        try:
            subprocess.check_call([editor, fname])
        except subprocess.CalledProcessError as e:
            raise PipError(
                "Editor Subprocess exited with exit code {}"
                .format(e.returncode)
            )
    def _get_n_args(self, args, example, n):
        if len(args) != n:
            msg = (
                'Got unexpected number of arguments, expected {}. '
                '(example: "{} config {}")'
            ).format(n, get_prog(), example)
            raise PipError(msg)
        if n == 1:
            return args[0]
        else:
            return args
    def _save_configuration(self):
        try:
            self.configuration.save()
        except Exception:
            logger.exception(
                "Unable to save configuration. Please report this as a bug."
            )
            raise PipError("Internal Error.")
    def _determine_editor(self, options):
        if options.editor is not None:
            return options.editor
        elif "VISUAL" in os.environ:
            return os.environ["VISUAL"]
        elif "EDITOR" in os.environ:
            return os.environ["EDITOR"]
        else:
            raise PipError("Could not determine editor to use.")
