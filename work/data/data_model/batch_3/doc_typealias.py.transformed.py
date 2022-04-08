
from typing import Dict
_EXTRA_DOCS: Dict[int, str] = {}
def document(obj, doc):
  try:
    obj.__doc__ = doc
  except AttributeError:
    _EXTRA_DOCS[id(obj)] = doc
