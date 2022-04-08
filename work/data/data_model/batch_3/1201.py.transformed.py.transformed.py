from __future__ import absolute_import, division, unicode_literals
import re
from . import base
from ..constants import rcdataElements, spaceCharacters
spaceCharacters = "".join(spaceCharacters)
SPACES_REGEX = re.compile("[%s]+" % spaceCharacters)
class Filter(base.Filter):
    spacePreserveElements = frozenset(["pre", "textarea"] + list(rcdataElements))
    def __iter__(self):
        preserve = 0
        for token in base.Filter.__iter__(self):
            type = token["type"]
            if type == "StartTag"                    and (preserve or token["name"] in self.spacePreserveElements):
                preserve += 1
            elif type == "EndTag" and preserve:
                preserve -= 1
            elif not preserve and type == "SpaceCharacters" and token["data"]:
                token["data"] = " "
            elif not preserve and type == "Characters":
                token["data"] = collapse_spaces(token["data"])
            yield token
def collapse_spaces(text):
    return SPACES_REGEX.sub(' ', text)