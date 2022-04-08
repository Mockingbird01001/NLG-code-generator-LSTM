
from wtforms import Form, validators, fields as f
from wtforms.compat import string_types
from wtforms.ext.appengine.fields import GeoPtPropertyField, KeyPropertyField, StringListPropertyField, IntegerListPropertyField
def get_TextField(kwargs):
    kwargs['validators'].append(validators.length(max=500))
    return f.TextField(**kwargs)
def get_IntegerField(kwargs):
    v = validators.NumberRange(min=-0x8000000000000000, max=0x7fffffffffffffff)
    kwargs['validators'].append(v)
    return f.IntegerField(**kwargs)
class ModelConverterBase(object):
    def __init__(self, converters=None):
        self.converters = {}
        for name in dir(self):
            if not name.startswith('convert_'):
                continue
            self.converters[name[8:]] = getattr(self, name)
    def convert(self, model, prop, field_args):
        prop_type_name = type(prop).__name__
        if(prop_type_name == "GenericProperty"):
            generic_type = field_args.get("type")
            if generic_type:
                prop_type_name = field_args.get("type")
        kwargs = {
            'label': prop._code_name.replace('_', ' ').title(),
            'default': prop._default,
            'validators': [],
        }
        if field_args:
            kwargs.update(field_args)
        if prop._required and prop_type_name not in self.NO_AUTO_REQUIRED:
            kwargs['validators'].append(validators.required())
        if kwargs.get('choices', None):
            kwargs['choices'] = [(v, v) for v in kwargs.get('choices')]
            return f.SelectField(**kwargs)
        if prop._choices:
            kwargs['choices'] = [(v, v) for v in prop._choices]
            return f.SelectField(**kwargs)
        else:
            converter = self.converters.get(prop_type_name, None)
            if converter is not None:
                return converter(model, prop, kwargs)
            else:
                return self.fallback_converter(model, prop, kwargs)
class ModelConverter(ModelConverterBase):
    NO_AUTO_REQUIRED = frozenset(['ListProperty', 'StringListProperty', 'BooleanProperty'])
    def convert_StringProperty(self, model, prop, kwargs):
        if prop._repeated:
            return StringListPropertyField(**kwargs)
        kwargs['validators'].append(validators.length(max=500))
        return get_TextField(kwargs)
    def convert_BooleanProperty(self, model, prop, kwargs):
        return f.BooleanField(**kwargs)
    def convert_IntegerProperty(self, model, prop, kwargs):
        if prop._repeated:
            return IntegerListPropertyField(**kwargs)
        return get_IntegerField(kwargs)
    def convert_FloatProperty(self, model, prop, kwargs):
        return f.FloatField(**kwargs)
    def convert_DateTimeProperty(self, model, prop, kwargs):
        if prop._auto_now or prop._auto_now_add:
            return None
        return f.DateTimeField(format='%Y-%m-%d %H:%M:%S', **kwargs)
    def convert_DateProperty(self, model, prop, kwargs):
        if prop._auto_now or prop._auto_now_add:
            return None
        return f.DateField(format='%Y-%m-%d', **kwargs)
    def convert_TimeProperty(self, model, prop, kwargs):
        if prop._auto_now or prop._auto_now_add:
            return None
        return f.DateTimeField(format='%H:%M:%S', **kwargs)
    def convert_RepeatedProperty(self, model, prop, kwargs):
        return None
    def convert_UserProperty(self, model, prop, kwargs):
        return None
    def convert_StructuredProperty(self, model, prop, kwargs):
        return None
    def convert_LocalStructuredProperty(self, model, prop, kwargs):
        return None
    def convert_JsonProperty(self, model, prop, kwargs):
        return None
    def convert_PickleProperty(self, model, prop, kwargs):
        return None
    def convert_GenericProperty(self, model, prop, kwargs):
        kwargs['validators'].append(validators.length(max=500))
        return get_TextField(kwargs)
    def convert_BlobKeyProperty(self, model, prop, kwargs):
        return f.FileField(**kwargs)
    def convert_TextProperty(self, model, prop, kwargs):
        return f.TextAreaField(**kwargs)
    def convert_ComputedProperty(self, model, prop, kwargs):
        return None
    def convert_GeoPtProperty(self, model, prop, kwargs):
        return GeoPtPropertyField(**kwargs)
    def convert_KeyProperty(self, model, prop, kwargs):
        if 'reference_class' not in kwargs:
            try:
                reference_class = prop._kind
            except AttributeError:
                reference_class = prop._reference_class
            if isinstance(reference_class, string_types):
                mod = __import__(model.__module__, None, None, [reference_class], 0)
                reference_class = getattr(mod, reference_class)
            kwargs['reference_class'] = reference_class
        kwargs.setdefault('allow_blank', not prop._required)
        return KeyPropertyField(**kwargs)
def model_fields(model, only=None, exclude=None, field_args=None,
                 converter=None):
    converter = converter or ModelConverter()
    field_args = field_args or {}
    props = model._properties
    field_names = list(x[0] for x in sorted(props.items(), key=lambda x: x[1]._creation_counter))
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
    return type(model._get_kind() + 'Form', (base_class,), field_dict)
