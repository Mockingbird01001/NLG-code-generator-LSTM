from typing import TYPE_CHECKING, Dict, Iterable, Iterator, Mapping, Sequence, Union
from pip._vendor.resolvelib.providers import AbstractProvider
from .base import Candidate, Constraint, Requirement
from .factory import Factory
if TYPE_CHECKING:
    from pip._vendor.resolvelib.providers import Preference
    from pip._vendor.resolvelib.resolvers import RequirementInformation
    PreferenceInformation = RequirementInformation[Requirement, Candidate]
    _ProviderBase = AbstractProvider[Requirement, Candidate, str]
else:
    _ProviderBase = AbstractProvider
class PipProvider(_ProviderBase):
    def __init__(
        self,
        factory,
        constraints,
        ignore_dependencies,
        upgrade_strategy,
        user_requested,
    ):
        self._factory = factory
        self._constraints = constraints
        self._ignore_dependencies = ignore_dependencies
        self._upgrade_strategy = upgrade_strategy
        self._user_requested = user_requested
    def identify(self, requirement_or_candidate):
        return requirement_or_candidate.name
    def get_preference(
        self,
        identifier: str,
        resolutions: Mapping[str, Candidate],
        candidates: Mapping[str, Iterator[Candidate]],
        information: Mapping[str, Iterator["PreferenceInformation"]],
    ) -> "Preference":
        def _get_restrictive_rating(requirements):
            lookups = (r.get_candidate_lookup() for r in requirements)
            cands, ireqs = zip(*lookups)
            if any(cand is not None for cand in cands):
                return 0
            spec_sets = (ireq.specifier for ireq in ireqs if ireq)
            operators = [
                specifier.operator for spec_set in spec_sets for specifier in spec_set
            ]
            if any(op in ("==", "===") for op in operators):
                return 1
            if operators:
                return 2
            return 3
        rating = _get_restrictive_rating(r for r, _ in information[identifier])
        order = self._user_requested.get(identifier, float("inf"))
        delay_this = identifier == "setuptools"
        return (delay_this, rating, order, identifier)
    def find_matches(
        self,
        identifier: str,
        requirements: Mapping[str, Iterator[Requirement]],
        incompatibilities: Mapping[str, Iterator[Candidate]],
    ) -> Iterable[Candidate]:
        def _eligible_for_upgrade(name):
            if self._upgrade_strategy == "eager":
                return True
            elif self._upgrade_strategy == "only-if-needed":
                return name in self._user_requested
            return False
        return self._factory.find_candidates(
            identifier=identifier,
            requirements=requirements,
            constraint=self._constraints.get(identifier, Constraint.empty()),
            prefers_installed=(not _eligible_for_upgrade(identifier)),
            incompatibilities=incompatibilities,
        )
    def is_satisfied_by(self, requirement, candidate):
        return requirement.is_satisfied_by(candidate)
    def get_dependencies(self, candidate):
        with_requires = not self._ignore_dependencies
        return [r for r in candidate.iter_dependencies(with_requires) if r is not None]
