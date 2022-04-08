
import json
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen
def assert_lower(string):
    assert string == string.lower()
    return string
def generate(url):
    parts = ['''\
LABELS = {
''']
    labels = [
        (repr(assert_lower(label)).lstrip('u'),
         repr(encoding['name']).lstrip('u'))
        for category in json.loads(urlopen(url).read().decode('ascii'))
        for encoding in category['encodings']
        for label in encoding['labels']]
    max_len = max(len(label) for label, name in labels)
    parts.extend(
        '    %s:%s %s,\n' % (label, ' ' * (max_len - len(label)), name)
        for label, name in labels)
    parts.append('}')
    return ''.join(parts)
if __name__ == '__main__':
    print(generate('http://encoding.spec.whatwg.org/encodings.json'))
