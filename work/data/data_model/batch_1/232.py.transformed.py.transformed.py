
from os.path import dirname
from code_generators.genapi import fullapi_hash
from code_generators.numpy_api import full_api
if __name__ == '__main__':
    curdir = dirname(__file__)
    print(fullapi_hash(full_api))
