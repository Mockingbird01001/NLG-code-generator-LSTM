import collections
import operator
from .providers import AbstractResolver
from .structs import DirectedGraph, IteratorMapping, build_iter_view
RequirementInformation = collections.namedtuple(
    "RequirementInformation", ["requirement", "parent"]
)
class ResolverException(Exception):
class RequirementsConflicted(ResolverException):
    def __init__(self, criterion):
        super(RequirementsConflicted, self).__init__(criterion)
        self.criterion = criterion
    def __str__(self):
        return "Requirements conflict: {}".format(
            ", ".join(repr(r) for r in self.criterion.iter_requirement()),
        )
class InconsistentCandidate(ResolverException):
    def __init__(self, candidate, criterion):
        super(InconsistentCandidate, self).__init__(candidate, criterion)
        self.candidate = candidate
        self.criterion = criterion
    def __str__(self):
        return "Provided candidate {!r} does not satisfy {}".format(
            self.candidate,
            ", ".join(repr(r) for r in self.criterion.iter_requirement()),
        )
class Criterion(object):
    def __init__(self, candidates, information, incompatibilities):
        self.candidates = candidates
        self.information = information
        self.incompatibilities = incompatibilities
    def __repr__(self):
        requirements = ", ".join(
            "({!r}, via={!r})".format(req, parent)
            for req, parent in self.information
        )
        return "Criterion({})".format(requirements)
    def iter_requirement(self):
        return (i.requirement for i in self.information)
    def iter_parent(self):
        return (i.parent for i in self.information)
class ResolutionError(ResolverException):
    pass
class ResolutionImpossible(ResolutionError):
    def __init__(self, causes):
        super(ResolutionImpossible, self).__init__(causes)
        self.causes = causes
class ResolutionTooDeep(ResolutionError):
    def __init__(self, round_count):
        super(ResolutionTooDeep, self).__init__(round_count)
        self.round_count = round_count
State = collections.namedtuple("State", "mapping criteria")
class Resolution(object):
    def __init__(self, provider, reporter):
        self._p = provider
        self._r = reporter
        self._states = []
    @property
    def state(self):
        try:
            return self._states[-1]
        except IndexError:
            raise AttributeError("state")
    def _push_new_state(self):
        base = self._states[-1]
        state = State(
            mapping=base.mapping.copy(),
            criteria=base.criteria.copy(),
        )
        self._states.append(state)
    def _merge_into_criterion(self, requirement, parent):
        self._r.adding_requirement(requirement=requirement, parent=parent)
        identifier = self._p.identify(requirement_or_candidate=requirement)
        criterion = self.state.criteria.get(identifier)
        if criterion:
            incompatibilities = list(criterion.incompatibilities)
        else:
            incompatibilities = []
        matches = self._p.find_matches(
            identifier=identifier,
            requirements=IteratorMapping(
                self.state.criteria,
                operator.methodcaller("iter_requirement"),
                {identifier: [requirement]},
            ),
            incompatibilities=IteratorMapping(
                self.state.criteria,
                operator.attrgetter("incompatibilities"),
                {identifier: incompatibilities},
            ),
        )
        if criterion:
            information = list(criterion.information)
            information.append(RequirementInformation(requirement, parent))
        else:
            information = [RequirementInformation(requirement, parent)]
        criterion = Criterion(
            candidates=build_iter_view(matches),
            information=information,
            incompatibilities=incompatibilities,
        )
        if not criterion.candidates:
            raise RequirementsConflicted(criterion)
        return identifier, criterion
    def _get_preference(self, name):
        return self._p.get_preference(
            identifier=name,
            resolutions=self.state.mapping,
            candidates=IteratorMapping(
                self.state.criteria,
                operator.attrgetter("candidates"),
            ),
            information=IteratorMapping(
                self.state.criteria,
                operator.attrgetter("information"),
            ),
        )
    def _is_current_pin_satisfying(self, name, criterion):
        try:
            current_pin = self.state.mapping[name]
        except KeyError:
            return False
        return all(
            self._p.is_satisfied_by(requirement=r, candidate=current_pin)
            for r in criterion.iter_requirement()
        )
    def _get_criteria_to_update(self, candidate):
        criteria = {}
        for r in self._p.get_dependencies(candidate=candidate):
            name, crit = self._merge_into_criterion(r, parent=candidate)
            criteria[name] = crit
        return criteria
    def _attempt_to_pin_criterion(self, name):
        criterion = self.state.criteria[name]
        causes = []
        for candidate in criterion.candidates:
            try:
                criteria = self._get_criteria_to_update(candidate)
            except RequirementsConflicted as e:
                causes.append(e.criterion)
                continue
            satisfied = all(
                self._p.is_satisfied_by(requirement=r, candidate=candidate)
                for r in criterion.iter_requirement()
            )
            if not satisfied:
                raise InconsistentCandidate(candidate, criterion)
            self._r.pinning(candidate=candidate)
            self.state.mapping.pop(name, None)
            self.state.mapping[name] = candidate
            self.state.criteria.update(criteria)
            return []
        return causes
    def _backtrack(self):
        while len(self._states) >= 3:
            del self._states[-1]
            broken_state = self._states.pop()
            name, candidate = broken_state.mapping.popitem()
            incompatibilities_from_broken = [
                (k, list(v.incompatibilities))
                for k, v in broken_state.criteria.items()
            ]
            incompatibilities_from_broken.append((name, [candidate]))
            self._r.backtracking(candidate=candidate)
            def _patch_criteria():
                for k, incompatibilities in incompatibilities_from_broken:
                    if not incompatibilities:
                        continue
                    try:
                        criterion = self.state.criteria[k]
                    except KeyError:
                        continue
                    matches = self._p.find_matches(
                        identifier=k,
                        requirements=IteratorMapping(
                            self.state.criteria,
                            operator.methodcaller("iter_requirement"),
                        ),
                        incompatibilities=IteratorMapping(
                            self.state.criteria,
                            operator.attrgetter("incompatibilities"),
                            {k: incompatibilities},
                        ),
                    )
                    candidates = build_iter_view(matches)
                    if not candidates:
                        return False
                    incompatibilities.extend(criterion.incompatibilities)
                    self.state.criteria[k] = Criterion(
                        candidates=candidates,
                        information=list(criterion.information),
                        incompatibilities=incompatibilities,
                    )
                return True
            self._push_new_state()
            success = _patch_criteria()
            if success:
                return True
        return False
    def resolve(self, requirements, max_rounds):
        if self._states:
            raise RuntimeError("already resolved")
        self._r.starting()
        self._states = [State(mapping=collections.OrderedDict(), criteria={})]
        for r in requirements:
            try:
                name, crit = self._merge_into_criterion(r, parent=None)
            except RequirementsConflicted as e:
                raise ResolutionImpossible(e.criterion.information)
            self.state.criteria[name] = crit
        self._push_new_state()
        for round_index in range(max_rounds):
            self._r.starting_round(index=round_index)
            unsatisfied_names = [
                key
                for key, criterion in self.state.criteria.items()
                if not self._is_current_pin_satisfying(key, criterion)
            ]
            if not unsatisfied_names:
                self._r.ending(state=self.state)
                return self.state
            name = min(unsatisfied_names, key=self._get_preference)
            failure_causes = self._attempt_to_pin_criterion(name)
            if failure_causes:
                success = self._backtrack()
                if not success:
                    causes = [i for c in failure_causes for i in c.information]
                    raise ResolutionImpossible(causes)
            else:
                self._push_new_state()
            self._r.ending_round(index=round_index, state=self.state)
        raise ResolutionTooDeep(max_rounds)
def _has_route_to_root(criteria, key, all_keys, connected):
    if key in connected:
        return True
    if key not in criteria:
        return False
    for p in criteria[key].iter_parent():
        try:
            pkey = all_keys[id(p)]
        except KeyError:
            continue
        if pkey in connected:
            connected.add(key)
            return True
        if _has_route_to_root(criteria, pkey, all_keys, connected):
            connected.add(key)
            return True
    return False
Result = collections.namedtuple("Result", "mapping graph criteria")
def _build_result(state):
    mapping = state.mapping
    all_keys = {id(v): k for k, v in mapping.items()}
    all_keys[id(None)] = None
    graph = DirectedGraph()
    graph.add(None)
    connected = {None}
    for key, criterion in state.criteria.items():
        if not _has_route_to_root(state.criteria, key, all_keys, connected):
            continue
        if key not in graph:
            graph.add(key)
        for p in criterion.iter_parent():
            try:
                pkey = all_keys[id(p)]
            except KeyError:
                continue
            if pkey not in graph:
                graph.add(pkey)
            graph.connect(pkey, key)
    return Result(
        mapping={k: v for k, v in mapping.items() if k in connected},
        graph=graph,
        criteria=state.criteria,
    )
class Resolver(AbstractResolver):
    base_exception = ResolverException
    def resolve(self, requirements, max_rounds=100):
        resolution = Resolution(self.provider, self.reporter)
        state = resolution.resolve(requirements, max_rounds=max_rounds)
        return _build_result(state)
