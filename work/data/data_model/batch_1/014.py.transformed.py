
from __future__ import absolute_import
import logging
import os
import sys
import optparse
import warnings
from pip import cmdoptions
from pip.index import PackageFinder
from pip.locations import running_under_virtualenv
from pip.download import PipSession
from pip.exceptions import (BadCommand, InstallationError, UninstallationError,
                            CommandError, PreviousBuildDirError)
from pip.compat import logging_dictConfig
from pip.baseparser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip.req import InstallRequirement, parse_requirements
from pip.status_codes import (
    SUCCESS, ERROR, UNKNOWN_ERROR, VIRTUALENV_NOT_FOUND,
    PREVIOUS_BUILD_DIR_ERROR,
)
from pip.utils import deprecation, get_prog, normalize_path
from pip.utils.logging import IndentingFormatter
from pip.utils.outdated import pip_version_check
__all__ = ['Command']
logger = logging.getLogger(__name__)
class Command(object):
    name = None
    usage = None
    hidden = False
    log_streams = ("ext://sys.stdout", "ext://sys.stderr")
    def __init__(self, isolated=False):
        parser_kw = {
            'usage': self.usage,
            'prog': '%s %s' % (get_prog(), self.name),
            'formatter': UpdatingDefaultsHelpFormatter(),
            'add_help_option': False,
            'name': self.name,
            'description': self.__doc__,
            'isolated': isolated,
        }
        self.parser = ConfigOptionParser(**parser_kw)
        optgroup_name = '%s Options' % self.name.capitalize()
        self.cmd_opts = optparse.OptionGroup(self.parser, optgroup_name)
        gen_opts = cmdoptions.make_option_group(
            cmdoptions.general_group,
            self.parser,
        )
        self.parser.add_option_group(gen_opts)
    def _build_session(self, options, retries=None, timeout=None):
        session = PipSession(
            cache=(
                normalize_path(os.path.join(options.cache_dir, "http"))
                if options.cache_dir else None
            ),
            retries=retries if retries is not None else options.retries,
            insecure_hosts=options.trusted_hosts,
        )
        if options.cert:
            session.verify = options.cert
        if options.client_cert:
            session.cert = options.client_cert
        if options.timeout or timeout:
            session.timeout = (
                timeout if timeout is not None else options.timeout
            )
        if options.proxy:
            session.proxies = {
                "http": options.proxy,
                "https": options.proxy,
            }
        session.auth.prompting = not options.no_input
        return session
    def parse_args(self, args):
        return self.parser.parse_args(args)
    def main(self, args):
        options, args = self.parse_args(args)
        if options.quiet:
            if options.quiet == 1:
                level = "WARNING"
            if options.quiet == 2:
                level = "ERROR"
            else:
                level = "CRITICAL"
        elif options.verbose:
            level = "DEBUG"
        else:
            level = "INFO"
        root_level = level
        if options.log:
            root_level = "DEBUG"
        logging_dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "exclude_warnings": {
                    "()": "pip.utils.logging.MaxLevelFilter",
                    "level": logging.WARNING,
                },
            },
            "formatters": {
                "indent": {
                    "()": IndentingFormatter,
                    "format": "%(message)s",
                },
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "pip.utils.logging.ColorizedStreamHandler",
                    "stream": self.log_streams[0],
                    "filters": ["exclude_warnings"],
                    "formatter": "indent",
                },
                "console_errors": {
                    "level": "WARNING",
                    "class": "pip.utils.logging.ColorizedStreamHandler",
                    "stream": self.log_streams[1],
                    "formatter": "indent",
                },
                "user_log": {
                    "level": "DEBUG",
                    "class": "pip.utils.logging.BetterRotatingFileHandler",
                    "filename": options.log or "/dev/null",
                    "delay": True,
                    "formatter": "indent",
                },
            },
            "root": {
                "level": root_level,
                "handlers": list(filter(None, [
                    "console",
                    "console_errors",
                    "user_log" if options.log else None,
                ])),
            },
            "loggers": dict(
                (
                    name,
                    {
                        "level": (
                            "WARNING"
                            if level in ["INFO", "ERROR"]
                            else "DEBUG"
                        ),
                    },
                )
                for name in ["pip._vendor", "distlib", "requests", "urllib3"]
            ),
        })
        if sys.version_info[:2] == (2, 6):
            warnings.warn(
                "Python 2.6 is no longer supported by the Python core team, "
                "please upgrade your Python. A future version of pip will "
                "drop support for Python 2.6",
                deprecation.Python26DeprecationWarning
            )
        if options.no_input:
            os.environ['PIP_NO_INPUT'] = '1'
        if options.exists_action:
            os.environ['PIP_EXISTS_ACTION'] = ' '.join(options.exists_action)
        if options.require_venv:
            if not running_under_virtualenv():
                logger.critical(
                    'Could not find an activated virtualenv (required).'
                )
                sys.exit(VIRTUALENV_NOT_FOUND)
        try:
            status = self.run(options, args)
            if isinstance(status, int):
                return status
        except PreviousBuildDirError as exc:
            logger.critical(str(exc))
            logger.debug('Exception information:', exc_info=True)
            return PREVIOUS_BUILD_DIR_ERROR
        except (InstallationError, UninstallationError, BadCommand) as exc:
            logger.critical(str(exc))
            logger.debug('Exception information:', exc_info=True)
            return ERROR
        except CommandError as exc:
            logger.critical('ERROR: %s', exc)
            logger.debug('Exception information:', exc_info=True)
            return ERROR
        except KeyboardInterrupt:
            logger.critical('Operation cancelled by user')
            logger.debug('Exception information:', exc_info=True)
            return ERROR
        except:
            logger.critical('Exception:', exc_info=True)
            return UNKNOWN_ERROR
        finally:
            if (not options.disable_pip_version_check and not
                    getattr(options, "no_index", False)):
                with self._build_session(
                        options,
                        retries=0,
                        timeout=min(5, options.timeout)) as session:
                    pip_version_check(session)
        return SUCCESS
class RequirementCommand(Command):
    @staticmethod
    def populate_requirement_set(requirement_set, args, options, finder,
                                 session, name, wheel_cache):
        for filename in options.constraints:
            for req in parse_requirements(
                    filename,
                    constraint=True, finder=finder, options=options,
                    session=session, wheel_cache=wheel_cache):
                requirement_set.add_requirement(req)
        for req in args:
            requirement_set.add_requirement(
                InstallRequirement.from_line(
                    req, None, isolated=options.isolated_mode,
                    wheel_cache=wheel_cache
                )
            )
        for req in options.editables:
            requirement_set.add_requirement(
                InstallRequirement.from_editable(
                    req,
                    default_vcs=options.default_vcs,
                    isolated=options.isolated_mode,
                    wheel_cache=wheel_cache
                )
            )
        found_req_in_file = False
        for filename in options.requirements:
            for req in parse_requirements(
                    filename,
                    finder=finder, options=options, session=session,
                    wheel_cache=wheel_cache):
                found_req_in_file = True
                requirement_set.add_requirement(req)
        requirement_set.require_hashes = options.require_hashes
        if not (args or options.editables or found_req_in_file):
            opts = {'name': name}
            if options.find_links:
                msg = ('You must give at least one requirement to '
                       '%(name)s (maybe you meant "pip %(name)s '
                       '%(links)s"?)' %
                       dict(opts, links=' '.join(options.find_links)))
            else:
                msg = ('You must give at least one requirement '
                       'to %(name)s (see "pip help %(name)s")' % opts)
            logger.warning(msg)
    def _build_package_finder(self, options, session,
                              platform=None, python_versions=None,
                              abi=None, implementation=None):
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index:
            logger.debug('Ignoring indexes: %s', ','.join(index_urls))
            index_urls = []
        return PackageFinder(
            find_links=options.find_links,
            format_control=options.format_control,
            index_urls=index_urls,
            trusted_hosts=options.trusted_hosts,
            allow_all_prereleases=options.pre,
            process_dependency_links=options.process_dependency_links,
            session=session,
            platform=platform,
            versions=python_versions,
            abi=abi,
            implementation=implementation,
        )
