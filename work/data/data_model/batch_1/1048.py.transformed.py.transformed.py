
_all_chars = [chr(_m) for _m in range(256)]
_ascii_upper = _all_chars[65:65+26]
_ascii_lower = _all_chars[97:97+26]
LOWER_TABLE = "".join(_all_chars[:65] + _ascii_lower + _all_chars[65+26:])
UPPER_TABLE = "".join(_all_chars[:97] + _ascii_upper + _all_chars[97+26:])
def english_lower(s):
    lowered = s.translate(LOWER_TABLE)
    return lowered
def english_upper(s):
    uppered = s.translate(UPPER_TABLE)
    return uppered
def english_capitalize(s):
    if s:
        return english_upper(s[0]) + s[1:]
    else:
        return s
