
from wtforms import Form, validators, widgets, fields as f
from wtforms.compat import iteritems
from wtforms.ext.appengine.fields import GeoPtPropertyField, ReferencePropertyField, StringListPropertyField
def get_TextField(kwargs):
    kwargs['validators'].append(validators.length(max=500))
    return f.TextField(**kwargs)
def get_IntegerField(kwargs):
    v = validators.NumberRange(min=-0x8000000000000000, max=0x7fffffffffffffff)
    kwargs['validators'].append(v)
    return f.IntegerField(**kwargs)
def convert_StringProperty(model, prop, kwargs):
    if prop.multiline:
        kwargs['validators'].append(validators.length(max=500))
        return f.TextAreaField(**kwargs)
    else:
        return get_TextField(kwargs)
def convert_ByteStringProperty(model, prop, kwargs):
    return get_TextField(kwargs)
def convert_BooleanProperty(model, prop, kwargs):
    return f.BooleanField(**kwargs)
def convert_IntegerProperty(model, prop, kwargs):
    return get_IntegerField(kwargs)
def convert_FloatProperty(model, prop, kwargs):
    return f.FloatField(**kwargs)
def convert_DateTimeProperty(model, prop, kwargs):
    if prop.auto_now or prop.auto_now_add:
        return None
    kwargs.setdefault('format', '%Y-%m-%d %H:%M:%S')
    return f.DateTimeField(**kwargs)
def convert_DateProperty(model, prop, kwargs):
    if prop.auto_now or prop.auto_now_add:
        return None
    kwargs.setdefault('format', '%Y-%m-%d')
    return f.DateField(**kwargs)
def convert_TimeProperty(model, prop, kwargs):
    if prop.auto_now or prop.auto_now_add:
        return None
    kwargs.setdefault('format', '%H:%M:%S')
    return f.DateTimeField(**kwargs)
def convert_ListProperty(model, prop, kwargs):
    return None
def convert_StringListProperty(model, prop, kwargs):
    return StringListPropertyField(**kwargs)
def convert_ReferenceProperty(model, prop, kwargs):
    kwargs['reference_class'] = prop.reference_class
    kwargs.setdefault('allow_blank', not prop.required)
    return ReferencePropertyField(**kwargs)
def convert_SelfReferenceProperty(model, prop, kwargs):
    return None
def convert_UserProperty(model, prop, kwargs):
    return None
def convert_BlobProperty(model, prop, kwargs):
    return f.FileField(**kwargs)
def convert_TextProperty(model, prop, kwargs):
    return f.TextAreaField(**kwargs)
def convert_CategoryProperty(model, prop, kwargs):
    return get_TextField(kwargs)
def convert_LinkProperty(model, prop, kwargs):
    kwargs['validators'].append(validators.url())
    return get_TextField(kwargs)
def convert_EmailProperty(model, prop, kwargs):
    kwargs['validators'].append(validators.email())
    return get_TextField(kwargs)
def convert_GeoPtProperty(model, prop, kwargs):
    return GeoPtPropertyField(**kwargs)
def convert_IMProperty(model, prop, kwargs):
    return None
def convert_PhoneNumberProperty(model, prop, kwargs):
    return get_TextField(kwargs)
def convert_PostalAddressProperty(model, prop, kwargs):
    return get_TextField(kwargs)
def convert_RatingProperty(model, prop, kwargs):
    kwargs['validators'].append(validators.NumberRange(min=0, max=100))
    return f.IntegerField(**kwargs)
class ModelConverter(object):
    default_converters = {
        'StringProperty':        convert_StringProperty,
        'ByteStringProperty':    convert_ByteStringProperty,
        'BooleanProperty':       convert_BooleanProperty,
        'IntegerProperty':       convert_IntegerProperty,
        'FloatProperty':         convert_FloatProperty,
        'DateTimeProperty':      convert_DateTimeProperty,
        'DateProperty':          convert_DateProperty,
        'TimeProperty':          convert_TimeProperty,
        'ListProperty':          convert_ListProperty,
        'StringListProperty':    convert_StringListProperty,
        'ReferenceProperty':     convert_ReferenceProperty,
        'SelfReferenceProperty': convert_SelfReferenceProperty,
        'UserProperty':          convert_UserProperty,
        'BlobProperty':          convert_BlobProperty,
        'TextProperty':          convert_TextProperty,
        'CategoryProperty':      convert_CategoryProperty,
        'LinkProperty':          convert_LinkProperty,
        'EmailProperty':         convert_EmailProperty,
        'GeoPtProperty':         convert_GeoPtProperty,
        'IMProperty':            convert_IMProperty,
        'PhoneNumberProperty':   convert_PhoneNumberProperty,
        'PostalAddressProperty': convert_PostalAddressProperty,
        'RatingProperty':        convert_RatingProperty,
    }
    NO_AUTO_REQUIRED = frozenset(['ListProperty', 'StringListProperty', 'BooleanProperty'])
    def __init__(self, converters=None):
        self.converters = converters or self.default_converters
    def convert(self, model, prop, field_args):
        prop_type_name = type(prop).__name__
        kwargs = {
            'label': prop.name.replace('_', ' ').title(),
            'default': prop.default_value(),
            'validators': [],
        }
        if field_args:
            kwargs.update(field_args)
        if prop.required and prop_type_name not in self.NO_AUTO_REQUIRED:
            kwargs['validators'].append(validators.required())
        if prop.choices:
            if 'choices' not in kwargs:
                kwargs['choices'] = [(v, v) for v in prop.choices]
            return f.SelectField(**kwargs)
        else:
            converter = self.converters.get(prop_type_name, None)
            if converter is not None:
                return converter(model, prop, kwargs)
def model_fields(model, only=None, exclude=None, field_args=None,
                 converter=None):
    converter = converter or ModelConverter()
    field_args = field_args or {}
    props = model.properties()
    sorted_props = sorted(iteritems(props), key=lambda prop: prop[1].creation_counter)
    field_names = list(x[0] for x in sorted_props)
    if only:
        field_names = list(f for f in only if f in field_names)
    elif exclude:
        field_names = list(f for f in field_names if f not in exclude)
    field_dict = {}
    for name in field_names:
        field = converter.convert(model, props[name], field_args.get(name))
        if field is not None:
            field_dict[name] = field
    return field_dict
def model_form(model, base_class=Form, only=None, exclude=None, field_args=None,
               converter=None):
    field_dict = model_fields(model, only, exclude, field_args, converter)
    return type(model.kind() + 'Form', (base_class,), field_dict)
