from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._internal.req.req_install import InstallRequirement
from .base import Candidate, CandidateLookup, Requirement, format_name
class ExplicitRequirement(Requirement):
    def __init__(self, candidate):
        self.candidate = candidate
    def __str__(self):
        return str(self.candidate)
    def __repr__(self):
        return "{class_name}({candidate!r})".format(
            class_name=self.__class__.__name__,
            candidate=self.candidate,
        )
    @property
    def project_name(self):
        return self.candidate.project_name
    @property
    def name(self):
        return self.candidate.name
    def format_for_error(self):
        return self.candidate.format_for_error()
    def get_candidate_lookup(self):
        return self.candidate, None
    def is_satisfied_by(self, candidate):
        return candidate == self.candidate
class SpecifierRequirement(Requirement):
    def __init__(self, ireq):
        assert ireq.link is None, "This is a link, not a specifier"
        self._ireq = ireq
        self._extras = frozenset(ireq.extras)
    def __str__(self):
        return str(self._ireq.req)
    def __repr__(self):
        return "{class_name}({requirement!r})".format(
            class_name=self.__class__.__name__,
            requirement=str(self._ireq.req),
        )
    @property
    def project_name(self):
        assert self._ireq.req, "Specifier-backed ireq is always PEP 508"
        return canonicalize_name(self._ireq.req.name)
    @property
    def name(self):
        return format_name(self.project_name, self._extras)
    def format_for_error(self):
        parts = [s.strip() for s in str(self).split(",")]
        if len(parts) == 0:
            return ""
        elif len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    def get_candidate_lookup(self):
        return None, self._ireq
    def is_satisfied_by(self, candidate):
        assert candidate.name == self.name, (
            f"Internal issue: Candidate is not for this requirement "
            f"{candidate.name} vs {self.name}"
        )
        assert self._ireq.req, "Specifier-backed ireq is always PEP 508"
        spec = self._ireq.req.specifier
        return spec.contains(candidate.version, prereleases=True)
class RequiresPythonRequirement(Requirement):
    def __init__(self, specifier, match):
        self.specifier = specifier
        self._candidate = match
    def __str__(self):
        return f"Python {self.specifier}"
    def __repr__(self):
        return "{class_name}({specifier!r})".format(
            class_name=self.__class__.__name__,
            specifier=str(self.specifier),
        )
    @property
    def project_name(self):
        return self._candidate.project_name
    @property
    def name(self):
        return self._candidate.name
    def format_for_error(self):
        return str(self)
    def get_candidate_lookup(self):
        if self.specifier.contains(self._candidate.version, prereleases=True):
            return self._candidate, None
        return None, None
    def is_satisfied_by(self, candidate):
        assert candidate.name == self._candidate.name, "Not Python candidate"
        return self.specifier.contains(candidate.version, prereleases=True)
class UnsatisfiableRequirement(Requirement):
    def __init__(self, name):
        self._name = name
    def __str__(self):
        return f"{self._name} (unavailable)"
    def __repr__(self):
        return "{class_name}({name!r})".format(
            class_name=self.__class__.__name__,
            name=str(self._name),
        )
    @property
    def project_name(self):
        return self._name
    @property
    def name(self):
        return self._name
    def format_for_error(self):
        return str(self)
    def get_candidate_lookup(self):
        return None, None
    def is_satisfied_by(self, candidate):
        return False
