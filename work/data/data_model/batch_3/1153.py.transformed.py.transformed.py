from optparse import Values
from typing import List
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import SessionCommandMixin, warn_if_run_as_root
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import InstallationError
from pip._internal.req import parse_requirements
from pip._internal.req.constructors import (
    install_req_from_line,
    install_req_from_parsed_requirement,
)
from pip._internal.utils.misc import protect_pip_from_modification_on_windows
class UninstallCommand(Command, SessionCommandMixin):
    usage =
    def add_options(self):
        self.cmd_opts.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='file',
            help='Uninstall all the packages listed in the given requirements '
                 'file.  This option can be used multiple times.',
        )
        self.cmd_opts.add_option(
            '-y', '--yes',
            dest='yes',
            action='store_true',
            help="Don't ask for confirmation of uninstall deletions.")
        self.parser.insert_option_group(0, self.cmd_opts)
    def run(self, options, args):
        session = self.get_default_session(options)
        reqs_to_uninstall = {}
        for name in args:
            req = install_req_from_line(
                name, isolated=options.isolated_mode,
            )
            if req.name:
                reqs_to_uninstall[canonicalize_name(req.name)] = req
        for filename in options.requirements:
            for parsed_req in parse_requirements(
                    filename,
                    options=options,
                    session=session):
                req = install_req_from_parsed_requirement(
                    parsed_req,
                    isolated=options.isolated_mode
                )
                if req.name:
                    reqs_to_uninstall[canonicalize_name(req.name)] = req
        if not reqs_to_uninstall:
            raise InstallationError(
                f'You must give at least one requirement to {self.name} (see '
                f'"pip help {self.name}")'
            )
        protect_pip_from_modification_on_windows(
            modifying_pip="pip" in reqs_to_uninstall
        )
        for req in reqs_to_uninstall.values():
            uninstall_pathset = req.uninstall(
                auto_confirm=options.yes, verbose=self.verbosity > 0,
            )
            if uninstall_pathset:
                uninstall_pathset.commit()
        warn_if_run_as_root()
        return SUCCESS
