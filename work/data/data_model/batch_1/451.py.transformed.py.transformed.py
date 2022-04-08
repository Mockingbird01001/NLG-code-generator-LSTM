
from pygments.style import Style
from pygments.token import (Keyword, Name, Comment, String, Error,
                            Number, Operator, Generic, Whitespace,
                            Punctuation, Other, Literal)
class FlaskyStyle(Style):
    default_style = ""
    styles = {
        Comment.Preproc:           "noitalic",
    }
