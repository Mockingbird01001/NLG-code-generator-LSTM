
import sys
from string import punctuation
def remove_punctuation(content):
	content = content.translate(None, punctuation);
