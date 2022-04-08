
import sys
import locale
import pytest
__ALL__ = ['CommaDecimalPointLocale']
def find_comma_decimal_point_locale():
    if sys.platform == 'win32':
        locales = ['FRENCH']
    else:
        locales = ['fr_FR', 'fr_FR.UTF-8', 'fi_FI', 'fi_FI.UTF-8']
    old_locale = locale.getlocale(locale.LC_NUMERIC)
    new_locale = None
    try:
        for loc in locales:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                new_locale = loc
                break
            except locale.Error:
                pass
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=old_locale)
    return old_locale, new_locale
class CommaDecimalPointLocale:
    (cur_locale, tst_locale) = find_comma_decimal_point_locale()
    def setup(self):
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)
    def teardown(self):
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)
    def __enter__(self):
        if self.tst_locale is None:
            pytest.skip("No French locale available")
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)
    def __exit__(self, type, value, traceback):
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)
