import copy
from mongoengine.errors import InvalidQueryError
from mongoengine.queryset import transform
__all__ = ('Q',)
class QNodeVisitor(object):
    def visit_combination(self, combination):
        return combination
    def visit_query(self, query):
        return query
class DuplicateQueryConditionsError(InvalidQueryError):
    pass
class SimplificationVisitor(QNodeVisitor):
    def visit_combination(self, combination):
        if combination.operation == combination.AND:
            if all(isinstance(node, Q) for node in combination.children):
                queries = [n.query for n in combination.children]
                try:
                    return Q(**self._query_conjunction(queries))
                except DuplicateQueryConditionsError:
                    pass
        return combination
    def _query_conjunction(self, queries):
        query_ops = set()
        combined_query = {}
        for query in queries:
            ops = set(query.keys())
            intersection = ops.intersection(query_ops)
            if intersection:
                raise DuplicateQueryConditionsError()
            query_ops.update(ops)
            combined_query.update(copy.deepcopy(query))
        return combined_query
class QueryCompilerVisitor(QNodeVisitor):
    def __init__(self, document):
        self.document = document
    def visit_combination(self, combination):
        operator = '$and'
        if combination.operation == combination.OR:
            operator = '$or'
        return {operator: combination.children}
    def visit_query(self, query):
        return transform.query(self.document, **query.query)
class QNode(object):
    AND = 0
    OR = 1
    def to_query(self, document):
        query = self.accept(SimplificationVisitor())
        query = query.accept(QueryCompilerVisitor(document))
        return query
    def accept(self, visitor):
        raise NotImplementedError
    def _combine(self, other, operation):
        if getattr(other, 'empty', True):
            return self
        if self.empty:
            return other
        return QCombination(operation, [self, other])
    @property
    def empty(self):
        return False
    def __or__(self, other):
        return self._combine(other, self.OR)
    def __and__(self, other):
        return self._combine(other, self.AND)
class QCombination(QNode):
    def __init__(self, operation, children):
        self.operation = operation
        self.children = []
        for node in children:
            if isinstance(node, QCombination) and node.operation == operation:
                self.children += node.children
            else:
                self.children.append(node)
    def accept(self, visitor):
        for i in range(len(self.children)):
            if isinstance(self.children[i], QNode):
                self.children[i] = self.children[i].accept(visitor)
        return visitor.visit_combination(self)
    @property
    def empty(self):
        return not bool(self.children)
class Q(QNode):
    def __init__(self, **query):
        self.query = query
    def accept(self, visitor):
        return visitor.visit_query(self)
    @property
    def empty(self):
        return not bool(self.query)
