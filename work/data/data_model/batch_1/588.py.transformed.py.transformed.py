
import json
import os.path
from ..util import native, load_config_paths, save_config_path
class WheelKeys(object):
    SCHEMA = 1
    CONFIG_NAME = 'wheel.json'
    def __init__(self):
        self.data = {'signers': [], 'verifiers': []}
    def load(self):
        for path in load_config_paths('wheel'):
            conf = os.path.join(native(path), self.CONFIG_NAME)
            if os.path.exists(conf):
                with open(conf, 'r') as infile:
                    self.data = json.load(infile)
                    for x in ('signers', 'verifiers'):
                        if x not in self.data:
                            self.data[x] = []
                    if 'schema' not in self.data:
                        self.data['schema'] = self.SCHEMA
                    elif self.data['schema'] != self.SCHEMA:
                        raise ValueError(
                            "Bad wheel.json version {0}, expected {1}".format(
                                self.data['schema'], self.SCHEMA))
                break
        return self
    def save(self):
        path = save_config_path('wheel')
        conf = os.path.join(native(path), self.CONFIG_NAME)
        with open(conf, 'w+') as out:
            json.dump(self.data, out, indent=2)
        return self
    def trust(self, scope, vk):
        self.data['verifiers'].append({'scope': scope, 'vk': vk})
        return self
    def untrust(self, scope, vk):
        self.data['verifiers'].remove({'scope': scope, 'vk': vk})
        return self
    def trusted(self, scope=None):
        trust = [(x['scope'], x['vk']) for x in self.data['verifiers']
                 if x['scope'] in (scope, '+')]
        trust.sort(key=lambda x: x[0])
        trust.reverse()
        return trust
    def signers(self, scope):
        sign = [(x['scope'], x['vk']) for x in self.data['signers'] if x['scope'] in (scope, '+')]
        sign.sort(key=lambda x: x[0])
        sign.reverse()
        return sign
    def add_signer(self, scope, vk):
        self.data['signers'].append({'scope': scope, 'vk': vk})
