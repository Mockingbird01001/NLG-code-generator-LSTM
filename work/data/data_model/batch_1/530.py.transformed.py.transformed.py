from django.utils.translation import ugettext, ungettext
from wtforms import form
class DjangoTranslations(object):
    def gettext(self, string):
        return ugettext(string)
    def ngettext(self, singular, plural, n):
        return ungettext(singular, plural, n)
class Form(form.Form):
    _django_translations = DjangoTranslations()
    def _get_translations(self):
        return self._django_translations
