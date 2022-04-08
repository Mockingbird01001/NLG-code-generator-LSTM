from functools import partial
from mongoengine.queryset.queryset import QuerySet
__all__ = ('queryset_manager', 'QuerySetManager')
class QuerySetManager(object):
    get_queryset = None
    default = QuerySet
    def __init__(self, queryset_func=None):
        if queryset_func:
            self.get_queryset = queryset_func
    def __get__(self, instance, owner):
        if instance is not None:
            return self
        queryset_class = owner._meta.get('queryset_class', self.default)
        queryset = queryset_class(owner, owner._get_collection())
        if self.get_queryset:
            arg_count = self.get_queryset.func_code.co_argcount
            if arg_count == 1:
                queryset = self.get_queryset(queryset)
            elif arg_count == 2:
                queryset = self.get_queryset(owner, queryset)
            else:
                queryset = partial(self.get_queryset, owner, queryset)
        return queryset
def queryset_manager(func):
    return QuerySetManager(func)
