
from pymongo import common
class CollationStrength(object):
    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3
    QUATERNARY = 4
    IDENTICAL = 5
class CollationAlternate(object):
    NON_IGNORABLE = 'non-ignorable'
    SHIFTED = 'shifted'
class CollationMaxVariable(object):
    PUNCT = 'punct'
    SPACE = 'space'
class CollationCaseFirst(object):
    UPPER = 'upper'
    LOWER = 'lower'
    OFF = 'off'
class Collation(object):
    __slots__ = ("__document",)
    def __init__(self, locale,
                 caseLevel=None,
                 caseFirst=None,
                 strength=None,
                 numericOrdering=None,
                 alternate=None,
                 maxVariable=None,
                 normalization=None,
                 backwards=None,
                 **kwargs):
        locale = common.validate_string('locale', locale)
        self.__document = {'locale': locale}
        if caseLevel is not None:
            self.__document['caseLevel'] = common.validate_boolean(
                'caseLevel', caseLevel)
        if caseFirst is not None:
            self.__document['caseFirst'] = common.validate_string(
                'caseFirst', caseFirst)
        if strength is not None:
            self.__document['strength'] = common.validate_integer(
                'strength', strength)
        if numericOrdering is not None:
            self.__document['numericOrdering'] = common.validate_boolean(
                'numericOrdering', numericOrdering)
        if alternate is not None:
            self.__document['alternate'] = common.validate_string(
                'alternate', alternate)
        if maxVariable is not None:
            self.__document['maxVariable'] = common.validate_string(
                'maxVariable', maxVariable)
        if normalization is not None:
            self.__document['normalization'] = common.validate_boolean(
                'normalization', normalization)
        if backwards is not None:
            self.__document['backwards'] = common.validate_boolean(
                'backwards', backwards)
        self.__document.update(kwargs)
    @property
    def document(self):
        return self.__document.copy()
    def __repr__(self):
        document = self.document
        return 'Collation(%s)' % (
            ', '.join('%s=%r' % (key, document[key]) for key in document),)
    def __eq__(self, other):
        if isinstance(other, Collation):
            return self.document == other.document
        return NotImplemented
    def __ne__(self, other):
        return not self == other
def validate_collation_or_none(value):
    if value is None:
        return None
    if isinstance(value, Collation):
        return value.document
    if isinstance(value, dict):
        return value
    raise TypeError(
        'collation must be a dict, an instance of collation.Collation, '
        'or None.')
