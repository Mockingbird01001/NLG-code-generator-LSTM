
from functools import partial
from attr.exceptions import UnannotatedAttributeError
from . import setters
from ._make import NOTHING, _frozen_setattrs, attrib, attrs
def define(
    maybe_cls=None,
    *,
    these=None,
    repr=None,
    hash=None,
    init=None,
    slots=True,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=None,
    kw_only=False,
    cache_hash=False,
    auto_exc=True,
    eq=None,
    order=False,
    auto_detect=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
):
    def do_it(cls, auto_attribs):
        return attrs(
            maybe_cls=cls,
            these=these,
            repr=repr,
            hash=hash,
            init=init,
            slots=slots,
            frozen=frozen,
            weakref_slot=weakref_slot,
            str=str,
            auto_attribs=auto_attribs,
            kw_only=kw_only,
            cache_hash=cache_hash,
            auto_exc=auto_exc,
            eq=eq,
            order=order,
            auto_detect=auto_detect,
            collect_by_mro=True,
            getstate_setstate=getstate_setstate,
            on_setattr=on_setattr,
            field_transformer=field_transformer,
        )
    def wrap(cls):
        nonlocal frozen, on_setattr
        had_on_setattr = on_setattr not in (None, setters.NO_OP)
        if frozen is False and on_setattr is None:
            on_setattr = setters.validate
        for base_cls in cls.__bases__:
            if base_cls.__setattr__ is _frozen_setattrs:
                if had_on_setattr:
                    raise ValueError(
                        "Frozen classes can't use on_setattr "
                        "(frozen-ness was inherited)."
                    )
                on_setattr = setters.NO_OP
                break
        if auto_attribs is not None:
            return do_it(cls, auto_attribs)
        try:
            return do_it(cls, True)
        except UnannotatedAttributeError:
            return do_it(cls, False)
    if maybe_cls is None:
        return wrap
    else:
        return wrap(maybe_cls)
mutable = define
frozen = partial(define, frozen=True, on_setattr=None)
def field(
    *,
    default=NOTHING,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
):
    return attrib(
        default=default,
        validator=validator,
        repr=repr,
        hash=hash,
        init=init,
        metadata=metadata,
        converter=converter,
        factory=factory,
        kw_only=kw_only,
        eq=eq,
        order=order,
        on_setattr=on_setattr,
    )
