from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
baseChar =
letter = " | ".join([baseChar, ideographic])
name = " | ".join([letter, digit, ".", "-", "_", combiningCharacter,
                   extender])
nameFirst = " | ".join([letter, "_"])
def charStringToList(chars):
    charRanges = [item.strip() for item in chars.split(" | ")]
    rv = []
    for item in charRanges:
        foundMatch = False
        for regexp in (reChar, reCharRange):
            match = regexp.match(item)
            if match is not None:
                rv.append([hexToInt(item) for item in match.groups()])
                if len(rv[-1]) == 1:
                    rv[-1] = rv[-1] * 2
                foundMatch = True
                break
        if not foundMatch:
            assert len(item) == 1
            rv.append([ord(item)] * 2)
    rv = normaliseCharList(rv)
    return rv
def normaliseCharList(charList):
    charList = sorted(charList)
    for item in charList:
        assert item[1] >= item[0]
    rv = []
    i = 0
    while i < len(charList):
        j = 1
        rv.append(charList[i])
        while i + j < len(charList) and charList[i + j][0] <= rv[-1][1] + 1:
            rv[-1][1] = charList[i + j][1]
            j += 1
        i += j
    return rv
max_unicode = int("FFFF", 16)
def missingRanges(charList):
    rv = []
    if charList[0] != 0:
        rv.append([0, charList[0][0] - 1])
    for i, item in enumerate(charList[:-1]):
        rv.append([item[1] + 1, charList[i + 1][0] - 1])
    if charList[-1][1] != max_unicode:
        rv.append([charList[-1][1] + 1, max_unicode])
    return rv
def listToRegexpStr(charList):
    rv = []
    for item in charList:
        if item[0] == item[1]:
            rv.append(escapeRegexp(chr(item[0])))
        else:
            rv.append(escapeRegexp(chr(item[0])) + "-" +
                      escapeRegexp(chr(item[1])))
    return "[%s]" % "".join(rv)
def hexToInt(hex_str):
    return int(hex_str, 16)
def escapeRegexp(string):
    specialCharacters = (".", "^", "$", "*", "+", "?", "{", "}",
                         "[", "]", "|", "(", ")", "-")
    for char in specialCharacters:
        string = string.replace(char, "\\" + char)
    return string
nonXmlNameBMPRegexp = re.compile('[\x00-,/:-@\\[-\\^`\\{-\xb6\xb8-\xbf\xd7\xf7\u0132-\u0133\u013f-\u0140\u0149\u017f\u01c4-\u01cc\u01f1-\u01f3\u01f6-\u01f9\u0218-\u024f\u02a9-\u02ba\u02c2-\u02cf\u02d2-\u02ff\u0346-\u035f\u0362-\u0385\u038b\u038d\u03a2\u03cf\u03d7-\u03d9\u03db\u03dd\u03df\u03e1\u03f4-\u0400\u040d\u0450\u045d\u0482\u0487-\u048f\u04c5-\u04c6\u04c9-\u04ca\u04cd-\u04cf\u04ec-\u04ed\u04f6-\u04f7\u04fa-\u0530\u0557-\u0558\u055a-\u0560\u0587-\u0590\u05a2\u05ba\u05be\u05c0\u05c3\u05c5-\u05cf\u05eb-\u05ef\u05f3-\u0620\u063b-\u063f\u0653-\u065f\u066a-\u066f\u06b8-\u06b9\u06bf\u06cf\u06d4\u06e9\u06ee-\u06ef\u06fa-\u0900\u0904\u093a-\u093b\u094e-\u0950\u0955-\u0957\u0964-\u0965\u0970-\u0980\u0984\u098d-\u098e\u0991-\u0992\u09a9\u09b1\u09b3-\u09b5\u09ba-\u09bb\u09bd\u09c5-\u09c6\u09c9-\u09ca\u09ce-\u09d6\u09d8-\u09db\u09de\u09e4-\u09e5\u09f2-\u0a01\u0a03-\u0a04\u0a0b-\u0a0e\u0a11-\u0a12\u0a29\u0a31\u0a34\u0a37\u0a3a-\u0a3b\u0a3d\u0a43-\u0a46\u0a49-\u0a4a\u0a4e-\u0a58\u0a5d\u0a5f-\u0a65\u0a75-\u0a80\u0a84\u0a8c\u0a8e\u0a92\u0aa9\u0ab1\u0ab4\u0aba-\u0abb\u0ac6\u0aca\u0ace-\u0adf\u0ae1-\u0ae5\u0af0-\u0b00\u0b04\u0b0d-\u0b0e\u0b11-\u0b12\u0b29\u0b31\u0b34-\u0b35\u0b3a-\u0b3b\u0b44-\u0b46\u0b49-\u0b4a\u0b4e-\u0b55\u0b58-\u0b5b\u0b5e\u0b62-\u0b65\u0b70-\u0b81\u0b84\u0b8b-\u0b8d\u0b91\u0b96-\u0b98\u0b9b\u0b9d\u0ba0-\u0ba2\u0ba5-\u0ba7\u0bab-\u0bad\u0bb6\u0bba-\u0bbd\u0bc3-\u0bc5\u0bc9\u0bce-\u0bd6\u0bd8-\u0be6\u0bf0-\u0c00\u0c04\u0c0d\u0c11\u0c29\u0c34\u0c3a-\u0c3d\u0c45\u0c49\u0c4e-\u0c54\u0c57-\u0c5f\u0c62-\u0c65\u0c70-\u0c81\u0c84\u0c8d\u0c91\u0ca9\u0cb4\u0cba-\u0cbd\u0cc5\u0cc9\u0cce-\u0cd4\u0cd7-\u0cdd\u0cdf\u0ce2-\u0ce5\u0cf0-\u0d01\u0d04\u0d0d\u0d11\u0d29\u0d3a-\u0d3d\u0d44-\u0d45\u0d49\u0d4e-\u0d56\u0d58-\u0d5f\u0d62-\u0d65\u0d70-\u0e00\u0e2f\u0e3b-\u0e3f\u0e4f\u0e5a-\u0e80\u0e83\u0e85-\u0e86\u0e89\u0e8b-\u0e8c\u0e8e-\u0e93\u0e98\u0ea0\u0ea4\u0ea6\u0ea8-\u0ea9\u0eac\u0eaf\u0eba\u0ebe-\u0ebf\u0ec5\u0ec7\u0ece-\u0ecf\u0eda-\u0f17\u0f1a-\u0f1f\u0f2a-\u0f34\u0f36\u0f38\u0f3a-\u0f3d\u0f48\u0f6a-\u0f70\u0f85\u0f8c-\u0f8f\u0f96\u0f98\u0fae-\u0fb0\u0fb8\u0fba-\u109f\u10c6-\u10cf\u10f7-\u10ff\u1101\u1104\u1108\u110a\u110d\u1113-\u113b\u113d\u113f\u1141-\u114b\u114d\u114f\u1151-\u1153\u1156-\u1158\u115a-\u115e\u1162\u1164\u1166\u1168\u116a-\u116c\u116f-\u1171\u1174\u1176-\u119d\u119f-\u11a7\u11a9-\u11aa\u11ac-\u11ad\u11b0-\u11b6\u11b9\u11bb\u11c3-\u11ea\u11ec-\u11ef\u11f1-\u11f8\u11fa-\u1dff\u1e9c-\u1e9f\u1efa-\u1eff\u1f16-\u1f17\u1f1e-\u1f1f\u1f46-\u1f47\u1f4e-\u1f4f\u1f58\u1f5a\u1f5c\u1f5e\u1f7e-\u1f7f\u1fb5\u1fbd\u1fbf-\u1fc1\u1fc5\u1fcd-\u1fcf\u1fd4-\u1fd5\u1fdc-\u1fdf\u1fed-\u1ff1\u1ff5\u1ffd-\u20cf\u20dd-\u20e0\u20e2-\u2125\u2127-\u2129\u212c-\u212d\u212f-\u217f\u2183-\u3004\u3006\u3008-\u3020\u3030\u3036-\u3040\u3095-\u3098\u309b-\u309c\u309f-\u30a0\u30fb\u30ff-\u3104\u312d-\u4dff\u9fa6-\uabff\ud7a4-\uffff]')
nonXmlNameFirstBMPRegexp = re.compile('[\x00-@\\[-\\^`\\{-\xbf\xd7\xf7\u0132-\u0133\u013f-\u0140\u0149\u017f\u01c4-\u01cc\u01f1-\u01f3\u01f6-\u01f9\u0218-\u024f\u02a9-\u02ba\u02c2-\u0385\u0387\u038b\u038d\u03a2\u03cf\u03d7-\u03d9\u03db\u03dd\u03df\u03e1\u03f4-\u0400\u040d\u0450\u045d\u0482-\u048f\u04c5-\u04c6\u04c9-\u04ca\u04cd-\u04cf\u04ec-\u04ed\u04f6-\u04f7\u04fa-\u0530\u0557-\u0558\u055a-\u0560\u0587-\u05cf\u05eb-\u05ef\u05f3-\u0620\u063b-\u0640\u064b-\u0670\u06b8-\u06b9\u06bf\u06cf\u06d4\u06d6-\u06e4\u06e7-\u0904\u093a-\u093c\u093e-\u0957\u0962-\u0984\u098d-\u098e\u0991-\u0992\u09a9\u09b1\u09b3-\u09b5\u09ba-\u09db\u09de\u09e2-\u09ef\u09f2-\u0a04\u0a0b-\u0a0e\u0a11-\u0a12\u0a29\u0a31\u0a34\u0a37\u0a3a-\u0a58\u0a5d\u0a5f-\u0a71\u0a75-\u0a84\u0a8c\u0a8e\u0a92\u0aa9\u0ab1\u0ab4\u0aba-\u0abc\u0abe-\u0adf\u0ae1-\u0b04\u0b0d-\u0b0e\u0b11-\u0b12\u0b29\u0b31\u0b34-\u0b35\u0b3a-\u0b3c\u0b3e-\u0b5b\u0b5e\u0b62-\u0b84\u0b8b-\u0b8d\u0b91\u0b96-\u0b98\u0b9b\u0b9d\u0ba0-\u0ba2\u0ba5-\u0ba7\u0bab-\u0bad\u0bb6\u0bba-\u0c04\u0c0d\u0c11\u0c29\u0c34\u0c3a-\u0c5f\u0c62-\u0c84\u0c8d\u0c91\u0ca9\u0cb4\u0cba-\u0cdd\u0cdf\u0ce2-\u0d04\u0d0d\u0d11\u0d29\u0d3a-\u0d5f\u0d62-\u0e00\u0e2f\u0e31\u0e34-\u0e3f\u0e46-\u0e80\u0e83\u0e85-\u0e86\u0e89\u0e8b-\u0e8c\u0e8e-\u0e93\u0e98\u0ea0\u0ea4\u0ea6\u0ea8-\u0ea9\u0eac\u0eaf\u0eb1\u0eb4-\u0ebc\u0ebe-\u0ebf\u0ec5-\u0f3f\u0f48\u0f6a-\u109f\u10c6-\u10cf\u10f7-\u10ff\u1101\u1104\u1108\u110a\u110d\u1113-\u113b\u113d\u113f\u1141-\u114b\u114d\u114f\u1151-\u1153\u1156-\u1158\u115a-\u115e\u1162\u1164\u1166\u1168\u116a-\u116c\u116f-\u1171\u1174\u1176-\u119d\u119f-\u11a7\u11a9-\u11aa\u11ac-\u11ad\u11b0-\u11b6\u11b9\u11bb\u11c3-\u11ea\u11ec-\u11ef\u11f1-\u11f8\u11fa-\u1dff\u1e9c-\u1e9f\u1efa-\u1eff\u1f16-\u1f17\u1f1e-\u1f1f\u1f46-\u1f47\u1f4e-\u1f4f\u1f58\u1f5a\u1f5c\u1f5e\u1f7e-\u1f7f\u1fb5\u1fbd\u1fbf-\u1fc1\u1fc5\u1fcd-\u1fcf\u1fd4-\u1fd5\u1fdc-\u1fdf\u1fed-\u1ff1\u1ff5\u1ffd-\u2125\u2127-\u2129\u212c-\u212d\u212f-\u217f\u2183-\u3006\u3008-\u3020\u302a-\u3040\u3095-\u30a0\u30fb-\u3104\u312d-\u4dff\u9fa6-\uabff\ud7a4-\uffff]')
class InfosetFilter(object):
    replacementRegexp = re.compile(r"U[\dA-F]{5,5}")
    def __init__(self,
                 dropXmlnsLocalName=False,
                 dropXmlnsAttrNs=False,
                 preventDoubleDashComments=False,
                 preventDashAtCommentEnd=False,
                 replaceFormFeedCharacters=True,
                 preventSingleQuotePubid=False):
        self.dropXmlnsLocalName = dropXmlnsLocalName
        self.dropXmlnsAttrNs = dropXmlnsAttrNs
        self.preventDoubleDashComments = preventDoubleDashComments
        self.preventDashAtCommentEnd = preventDashAtCommentEnd
        self.replaceFormFeedCharacters = replaceFormFeedCharacters
        self.preventSingleQuotePubid = preventSingleQuotePubid
        self.replaceCache = {}
    def coerceAttribute(self, name, namespace=None):
        if self.dropXmlnsLocalName and name.startswith("xmlns:"):
            warnings.warn("Attributes cannot begin with xmlns", DataLossWarning)
            return None
        elif (self.dropXmlnsAttrNs and
              namespace == "http://www.w3.org/2000/xmlns/"):
            warnings.warn("Attributes cannot be in the xml namespace", DataLossWarning)
            return None
        else:
            return self.toXmlName(name)
    def coerceElement(self, name):
        return self.toXmlName(name)
    def coerceComment(self, data):
        if self.preventDoubleDashComments:
            while "--" in data:
                warnings.warn("Comments cannot contain adjacent dashes", DataLossWarning)
                data = data.replace("--", "- -")
            if data.endswith("-"):
                warnings.warn("Comments cannot end in a dash", DataLossWarning)
                data += " "
        return data
    def coerceCharacters(self, data):
        if self.replaceFormFeedCharacters:
            for _ in range(data.count("\x0C")):
                warnings.warn("Text cannot contain U+000C", DataLossWarning)
            data = data.replace("\x0C", " ")
        return data
    def coercePubid(self, data):
        dataOutput = data
        for char in nonPubidCharRegexp.findall(data):
            warnings.warn("Coercing non-XML pubid", DataLossWarning)
            replacement = self.getReplacementCharacter(char)
            dataOutput = dataOutput.replace(char, replacement)
        if self.preventSingleQuotePubid and dataOutput.find("'") >= 0:
            warnings.warn("Pubid cannot contain single quote", DataLossWarning)
            dataOutput = dataOutput.replace("'", self.getReplacementCharacter("'"))
        return dataOutput
    def toXmlName(self, name):
        nameFirst = name[0]
        nameRest = name[1:]
        m = nonXmlNameFirstBMPRegexp.match(nameFirst)
        if m:
            warnings.warn("Coercing non-XML name: %s" % name, DataLossWarning)
            nameFirstOutput = self.getReplacementCharacter(nameFirst)
        else:
            nameFirstOutput = nameFirst
        nameRestOutput = nameRest
        replaceChars = set(nonXmlNameBMPRegexp.findall(nameRest))
        for char in replaceChars:
            warnings.warn("Coercing non-XML name: %s" % name, DataLossWarning)
            replacement = self.getReplacementCharacter(char)
            nameRestOutput = nameRestOutput.replace(char, replacement)
        return nameFirstOutput + nameRestOutput
    def getReplacementCharacter(self, char):
        if char in self.replaceCache:
            replacement = self.replaceCache[char]
        else:
            replacement = self.escapeChar(char)
        return replacement
    def fromXmlName(self, name):
        for item in set(self.replacementRegexp.findall(name)):
            name = name.replace(item, self.unescapeChar(item))
        return name
    def escapeChar(self, char):
        replacement = "U%05X" % ord(char)
        self.replaceCache[char] = replacement
        return replacement
    def unescapeChar(self, charcode):
        return chr(int(charcode[1:], 16))
