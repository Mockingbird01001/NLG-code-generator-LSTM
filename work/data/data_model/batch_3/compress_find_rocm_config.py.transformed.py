
import base64
import zlib
def main():
  with open('find_rocm_config.py', 'rb') as f:
    data = f.read()
  compressed = zlib.compress(data)
  b64encoded = base64.b64encode(compressed)
  with open('find_rocm_config.py.gz.base64', 'wb') as f:
    f.write(b64encoded)
if __name__ == '__main__':
  main()
