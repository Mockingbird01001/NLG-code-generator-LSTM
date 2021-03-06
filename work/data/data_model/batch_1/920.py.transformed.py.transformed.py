import logging
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import InstallationError
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils import compatibility_tags
logger = logging.getLogger(__name__)
class RequirementSet:
    def __init__(self, check_supported_wheels=True):
        self.requirements = OrderedDict()
        self.check_supported_wheels = check_supported_wheels
        self.unnamed_requirements = []
    def __str__(self):
        requirements = sorted(
            (req for req in self.requirements.values() if not req.comes_from),
            key=lambda req: canonicalize_name(req.name or ""),
        )
        return ' '.join(str(req.req) for req in requirements)
    def __repr__(self):
        requirements = sorted(
            self.requirements.values(),
            key=lambda req: canonicalize_name(req.name or ""),
        )
        format_string = '<{classname} object; {count} requirement(s): {reqs}>'
        return format_string.format(
            classname=self.__class__.__name__,
            count=len(requirements),
            reqs=', '.join(str(req.req) for req in requirements),
        )
    def add_unnamed_requirement(self, install_req):
        assert not install_req.name
        self.unnamed_requirements.append(install_req)
    def add_named_requirement(self, install_req):
        assert install_req.name
        project_name = canonicalize_name(install_req.name)
        self.requirements[project_name] = install_req
    def add_requirement(
        self,
        install_req,
        parent_req_name=None,
        extras_requested=None
    ):
        if not install_req.match_markers(extras_requested):
            logger.info(
                "Ignoring %s: markers '%s' don't match your environment",
                install_req.name, install_req.markers,
            )
            return [], None
        if install_req.link and install_req.link.is_wheel:
            wheel = Wheel(install_req.link.filename)
            tags = compatibility_tags.get_supported()
            if (self.check_supported_wheels and not wheel.supported(tags)):
                raise InstallationError(
                    "{} is not a supported wheel on this platform.".format(
                        wheel.filename)
                )
        assert not install_req.user_supplied or parent_req_name is None, (
            "a user supplied req shouldn't have a parent"
        )
        if not install_req.name:
            self.add_unnamed_requirement(install_req)
            return [install_req], None
        try:
            existing_req = self.get_requirement(
                install_req.name)
        except KeyError:
            existing_req = None
        has_conflicting_requirement = (
            parent_req_name is None and
            existing_req and
            not existing_req.constraint and
            existing_req.extras == install_req.extras and
            existing_req.req and
            install_req.req and
            existing_req.req.specifier != install_req.req.specifier
        )
        if has_conflicting_requirement:
            raise InstallationError(
                "Double requirement given: {} (already in {}, name={!r})"
                .format(install_req, existing_req, install_req.name)
            )
        if not existing_req:
            self.add_named_requirement(install_req)
            return [install_req], install_req
        if install_req.constraint or not existing_req.constraint:
            return [], existing_req
        does_not_satisfy_constraint = (
            install_req.link and
            not (
                existing_req.link and
                install_req.link.path == existing_req.link.path
            )
        )
        if does_not_satisfy_constraint:
            raise InstallationError(
                "Could not satisfy constraints for '{}': "
                "installation from path or url cannot be "
                "constrained to a version".format(install_req.name)
            )
        existing_req.constraint = False
        if install_req.user_supplied:
            existing_req.user_supplied = True
        existing_req.extras = tuple(sorted(
            set(existing_req.extras) | set(install_req.extras)
        ))
        logger.debug(
            "Setting %s extras to: %s",
            existing_req, existing_req.extras,
        )
        return [existing_req], existing_req
    def has_requirement(self, name):
        project_name = canonicalize_name(name)
        return (
            project_name in self.requirements and
            not self.requirements[project_name].constraint
        )
    def get_requirement(self, name):
        project_name = canonicalize_name(name)
        if project_name in self.requirements:
            return self.requirements[project_name]
        raise KeyError(f"No project with the name {name!r}")
    @property
    def all_requirements(self):
        return self.unnamed_requirements + list(self.requirements.values())
