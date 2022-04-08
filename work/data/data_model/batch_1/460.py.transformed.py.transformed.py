import warnings
from wtforms import form
from wtforms.ext.i18n.utils import get_translations
translations_cache = {}
class Form(form.Form):
    LANGUAGES = None
    def __init__(self, *args, **kwargs):
        warnings.warn(
            'i18n is now in core, wtforms.ext.i18n will be removed in WTForms 3.0',
            DeprecationWarning, stacklevel=2
        )
        if 'LANGUAGES' in kwargs:
            self.LANGUAGES = kwargs.pop('LANGUAGES')
        super(Form, self).__init__(*args, **kwargs)
    def _get_translations(self):
        languages = tuple(self.LANGUAGES) if self.LANGUAGES else (self.meta.locales or None)
        if languages not in translations_cache:
            translations_cache[languages] = get_translations(languages)
        return translations_cache[languages]
