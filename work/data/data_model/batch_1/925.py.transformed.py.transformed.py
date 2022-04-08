import functools
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, cast
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.resolvelib import BaseReporter, ResolutionImpossible
from pip._vendor.resolvelib import Resolver as RLResolver
from pip._vendor.resolvelib.structs import DirectedGraph
from pip._internal.cache import WheelCache
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.req_install import (
    InstallRequirement,
    check_invalid_constraint_type,
)
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.resolution.resolvelib.provider import PipProvider
from pip._internal.resolution.resolvelib.reporter import (
    PipDebuggingReporter,
    PipReporter,
)
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import dist_is_editable
from .base import Candidate, Constraint, Requirement
from .factory import Factory
if TYPE_CHECKING:
    from pip._vendor.resolvelib.resolvers import Result as RLResult
    Result = RLResult[Requirement, Candidate, str]
logger = logging.getLogger(__name__)
class Resolver(BaseResolver):
    _allowed_strategies = {"eager", "only-if-needed", "to-satisfy-only"}
    def __init__(
        self,
        preparer,
        finder,
        wheel_cache,
        make_install_req,
        use_user_site,
        ignore_dependencies,
        ignore_installed,
        ignore_requires_python,
        force_reinstall,
        upgrade_strategy,
        py_version_info=None,
    ):
        super().__init__()
        assert upgrade_strategy in self._allowed_strategies
        self.factory = Factory(
            finder=finder,
            preparer=preparer,
            make_install_req=make_install_req,
            wheel_cache=wheel_cache,
            use_user_site=use_user_site,
            force_reinstall=force_reinstall,
            ignore_installed=ignore_installed,
            ignore_requires_python=ignore_requires_python,
            py_version_info=py_version_info,
        )
        self.ignore_dependencies = ignore_dependencies
        self.upgrade_strategy = upgrade_strategy
        self._result = None
    def resolve(self, root_reqs, check_supported_wheels):
        constraints = {}
        user_requested = {}
        requirements = []
        for i, req in enumerate(root_reqs):
            if req.constraint:
                problem = check_invalid_constraint_type(req)
                if problem:
                    raise InstallationError(problem)
                if not req.match_markers():
                    continue
                assert req.name, "Constraint must be named"
                name = canonicalize_name(req.name)
                if name in constraints:
                    constraints[name] &= req
                else:
                    constraints[name] = Constraint.from_ireq(req)
            else:
                if req.user_supplied and req.name:
                    canonical_name = canonicalize_name(req.name)
                    if canonical_name not in user_requested:
                        user_requested[canonical_name] = i
                r = self.factory.make_requirement_from_install_req(
                    req, requested_extras=()
                )
                if r is not None:
                    requirements.append(r)
        provider = PipProvider(
            factory=self.factory,
            constraints=constraints,
            ignore_dependencies=self.ignore_dependencies,
            upgrade_strategy=self.upgrade_strategy,
            user_requested=user_requested,
        )
        if "PIP_RESOLVER_DEBUG" in os.environ:
            reporter = PipDebuggingReporter()
        else:
            reporter = PipReporter()
        resolver = RLResolver(
            provider,
            reporter,
        )
        try:
            try_to_avoid_resolution_too_deep = 2000000
            result = self._result = resolver.resolve(
                requirements, max_rounds=try_to_avoid_resolution_too_deep
            )
        except ResolutionImpossible as e:
            error = self.factory.get_installation_error(
                cast("ResolutionImpossible[Requirement, Candidate]", e),
                constraints,
            )
            raise error from e
        req_set = RequirementSet(check_supported_wheels=check_supported_wheels)
        for candidate in result.mapping.values():
            ireq = candidate.get_install_requirement()
            if ireq is None:
                continue
            installed_dist = self.factory.get_dist_to_uninstall(candidate)
            if installed_dist is None:
                ireq.should_reinstall = False
            elif self.factory.force_reinstall:
                ireq.should_reinstall = True
            elif parse_version(installed_dist.version) != candidate.version:
                ireq.should_reinstall = True
            elif candidate.is_editable or dist_is_editable(installed_dist):
                ireq.should_reinstall = True
            elif candidate.source_link and candidate.source_link.is_file:
                if candidate.source_link.is_wheel:
                    logger.info(
                        "%s is already installed with the same version as the "
                        "provided wheel. Use --force-reinstall to force an "
                        "installation of the wheel.",
                        ireq.name,
                    )
                    continue
                looks_like_sdist = (
                    is_archive_file(candidate.source_link.file_path)
                    and candidate.source_link.ext != ".zip"
                )
                if looks_like_sdist:
                    reason = (
                        "Source distribution is being reinstalled despite an "
                        "installed package having the same name and version as "
                        "the installed package."
                    )
                    replacement = "use --force-reinstall"
                    deprecated(
                        reason=reason,
                        replacement=replacement,
                        gone_in="21.2",
                        issue=8711,
                    )
                ireq.should_reinstall = True
            else:
                continue
            link = candidate.source_link
            if link and link.is_yanked:
                msg = (
                    "The candidate selected for download or install is a "
                    "yanked version: {name!r} candidate (version {version} "
                    "at {link})\nReason for being yanked: {reason}"
                ).format(
                    name=candidate.name,
                    version=candidate.version,
                    link=link,
                    reason=link.yanked_reason or "<none given>",
                )
                logger.warning(msg)
            req_set.add_named_requirement(ireq)
        reqs = req_set.all_requirements
        self.factory.preparer.prepare_linked_requirements_more(reqs)
        return req_set
    def get_installation_order(self, req_set):
        assert self._result is not None, "must call resolve() first"
        graph = self._result.graph
        weights = get_topological_weights(
            graph,
            expected_node_count=len(self._result.mapping) + 1,
        )
        sorted_items = sorted(
            req_set.requirements.items(),
            key=functools.partial(_req_set_item_sorter, weights=weights),
            reverse=True,
        )
        return [ireq for _, ireq in sorted_items]
def get_topological_weights(graph, expected_node_count):
    path = set()
    weights = {}
    def visit(node):
        if node in path:
            return
        path.add(node)
        for child in graph.iter_children(node):
            visit(child)
        path.remove(node)
        last_known_parent_count = weights.get(node, 0)
        weights[node] = max(last_known_parent_count, len(path))
    visit(None)
    assert weights[None] == 0
    assert len(weights) == expected_node_count
    return weights
def _req_set_item_sorter(
    item,
    weights,
):
    name = canonicalize_name(item[0])
    return weights[name], name
