
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
try:
    from threading import Thread
except ImportError:
    from dummy_threading import Thread
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
                     urlparse, build_opener, string_types)
from .util import cached_property, zip_dir, ServerProxy
logger = logging.getLogger(__name__)
DEFAULT_INDEX = 'https://pypi.org/pypi'
DEFAULT_REALM = 'pypi'
class PackageIndex(object):
    boundary = b'----------ThIs_Is_tHe_distlib_index_bouNdaRY_$'
    def __init__(self, url=None):
        self.url = url or DEFAULT_INDEX
        self.read_configuration()
        scheme, netloc, path, params, query, frag = urlparse(self.url)
        if params or query or frag or scheme not in ('http', 'https'):
            raise DistlibException('invalid repository: %s' % self.url)
        self.password_handler = None
        self.ssl_verifier = None
        self.gpg = None
        self.gpg_home = None
        with open(os.devnull, 'w') as sink:
            for s in ('gpg', 'gpg2'):
                try:
                    rc = subprocess.check_call([s, '--version'], stdout=sink,
                                               stderr=sink)
                    if rc == 0:
                        self.gpg = s
                        break
                except OSError:
                    pass
    def _get_pypirc_command(self):
        from distutils.core import Distribution
        from distutils.config import PyPIRCCommand
        d = Distribution()
        return PyPIRCCommand(d)
    def read_configuration(self):
        c = self._get_pypirc_command()
        c.repository = self.url
        cfg = c._read_pypirc()
        self.username = cfg.get('username')
        self.password = cfg.get('password')
        self.realm = cfg.get('realm', 'pypi')
        self.url = cfg.get('repository', self.url)
    def save_configuration(self):
        self.check_credentials()
        c = self._get_pypirc_command()
        c._store_pypirc(self.username, self.password)
    def check_credentials(self):
        if self.username is None or self.password is None:
            raise DistlibException('username and password must be set')
        pm = HTTPPasswordMgr()
        _, netloc, _, _, _, _ = urlparse(self.url)
        pm.add_password(self.realm, netloc, self.username, self.password)
        self.password_handler = HTTPBasicAuthHandler(pm)
    def register(self, metadata):
        self.check_credentials()
        metadata.validate()
        d = metadata.todict()
        d[':action'] = 'verify'
        request = self.encode_request(d.items(), [])
        response = self.send_request(request)
        d[':action'] = 'submit'
        request = self.encode_request(d.items(), [])
        return self.send_request(request)
    def _reader(self, name, stream, outbuf):
        while True:
            s = stream.readline()
            if not s:
                break
            s = s.decode('utf-8').rstrip()
            outbuf.append(s)
            logger.debug('%s: %s' % (name, s))
        stream.close()
    def get_sign_command(self, filename, signer, sign_password,
                         keystore=None):
        cmd = [self.gpg, '--status-fd', '2', '--no-tty']
        if keystore is None:
            keystore = self.gpg_home
        if keystore:
            cmd.extend(['--homedir', keystore])
        if sign_password is not None:
            cmd.extend(['--batch', '--passphrase-fd', '0'])
        td = tempfile.mkdtemp()
        sf = os.path.join(td, os.path.basename(filename) + '.asc')
        cmd.extend(['--detach-sign', '--armor', '--local-user',
                    signer, '--output', sf, filename])
        logger.debug('invoking: %s', ' '.join(cmd))
        return cmd, sf
    def run_command(self, cmd, input_data=None):
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
        }
        if input_data is not None:
            kwargs['stdin'] = subprocess.PIPE
        stdout = []
        stderr = []
        p = subprocess.Popen(cmd, **kwargs)
        t1 = Thread(target=self._reader, args=('stdout', p.stdout, stdout))
        t1.start()
        t2 = Thread(target=self._reader, args=('stderr', p.stderr, stderr))
        t2.start()
        if input_data is not None:
            p.stdin.write(input_data)
            p.stdin.close()
        p.wait()
        t1.join()
        t2.join()
        return p.returncode, stdout, stderr
    def sign_file(self, filename, signer, sign_password, keystore=None):
        cmd, sig_file = self.get_sign_command(filename, signer, sign_password,
                                              keystore)
        rc, stdout, stderr = self.run_command(cmd,
                                              sign_password.encode('utf-8'))
        if rc != 0:
            raise DistlibException('sign command failed with error '
                                   'code %s' % rc)
        return sig_file
    def upload_file(self, metadata, filename, signer=None, sign_password=None,
                    filetype='sdist', pyversion='source', keystore=None):
        self.check_credentials()
        if not os.path.exists(filename):
            raise DistlibException('not found: %s' % filename)
        metadata.validate()
        d = metadata.todict()
        sig_file = None
        if signer:
            if not self.gpg:
                logger.warning('no signing program available - not signed')
            else:
                sig_file = self.sign_file(filename, signer, sign_password,
                                          keystore)
        with open(filename, 'rb') as f:
            file_data = f.read()
        md5_digest = hashlib.md5(file_data).hexdigest()
        sha256_digest = hashlib.sha256(file_data).hexdigest()
        d.update({
            ':action': 'file_upload',
            'protocol_version': '1',
            'filetype': filetype,
            'pyversion': pyversion,
            'md5_digest': md5_digest,
            'sha256_digest': sha256_digest,
        })
        files = [('content', os.path.basename(filename), file_data)]
        if sig_file:
            with open(sig_file, 'rb') as f:
                sig_data = f.read()
            files.append(('gpg_signature', os.path.basename(sig_file),
                         sig_data))
            shutil.rmtree(os.path.dirname(sig_file))
        request = self.encode_request(d.items(), files)
        return self.send_request(request)
    def upload_documentation(self, metadata, doc_dir):
        self.check_credentials()
        if not os.path.isdir(doc_dir):
            raise DistlibException('not a directory: %r' % doc_dir)
        fn = os.path.join(doc_dir, 'index.html')
        if not os.path.exists(fn):
            raise DistlibException('not found: %r' % fn)
        metadata.validate()
        name, version = metadata.name, metadata.version
        zip_data = zip_dir(doc_dir).getvalue()
        fields = [(':action', 'doc_upload'),
                  ('name', name), ('version', version)]
        files = [('content', name, zip_data)]
        request = self.encode_request(fields, files)
        return self.send_request(request)
    def get_verify_command(self, signature_filename, data_filename,
                           keystore=None):
        cmd = [self.gpg, '--status-fd', '2', '--no-tty']
        if keystore is None:
            keystore = self.gpg_home
        if keystore:
            cmd.extend(['--homedir', keystore])
        cmd.extend(['--verify', signature_filename, data_filename])
        logger.debug('invoking: %s', ' '.join(cmd))
        return cmd
    def verify_signature(self, signature_filename, data_filename,
                         keystore=None):
        if not self.gpg:
            raise DistlibException('verification unavailable because gpg '
                                   'unavailable')
        cmd = self.get_verify_command(signature_filename, data_filename,
                                      keystore)
        rc, stdout, stderr = self.run_command(cmd)
        if rc not in (0, 1):
            raise DistlibException('verify command failed with error '
                             'code %s' % rc)
        return rc == 0
    def download_file(self, url, destfile, digest=None, reporthook=None):
        if digest is None:
            digester = None
            logger.debug('No digest specified')
        else:
            if isinstance(digest, (list, tuple)):
                hasher, digest = digest
            else:
                hasher = 'md5'
            digester = getattr(hashlib, hasher)()
            logger.debug('Digest specified: %s' % digest)
        with open(destfile, 'wb') as dfp:
            sfp = self.send_request(Request(url))
            try:
                headers = sfp.info()
                blocksize = 8192
                size = -1
                read = 0
                blocknum = 0
                if "content-length" in headers:
                    size = int(headers["Content-Length"])
                if reporthook:
                    reporthook(blocknum, blocksize, size)
                while True:
                    block = sfp.read(blocksize)
                    if not block:
                        break
                    read += len(block)
                    dfp.write(block)
                    if digester:
                        digester.update(block)
                    blocknum += 1
                    if reporthook:
                        reporthook(blocknum, blocksize, size)
            finally:
                sfp.close()
        if size >= 0 and read < size:
            raise DistlibException(
                'retrieval incomplete: got only %d out of %d bytes'
                % (read, size))
        if digester:
            actual = digester.hexdigest()
            if digest != actual:
                raise DistlibException('%s digest mismatch for %s: expected '
                                       '%s, got %s' % (hasher, destfile,
                                                       digest, actual))
            logger.debug('Digest verified: %s', digest)
    def send_request(self, req):
        handlers = []
        if self.password_handler:
            handlers.append(self.password_handler)
        if self.ssl_verifier:
            handlers.append(self.ssl_verifier)
        opener = build_opener(*handlers)
        return opener.open(req)
    def encode_request(self, fields, files):
        parts = []
        boundary = self.boundary
        for k, values in fields:
            if not isinstance(values, (list, tuple)):
                values = [values]
            for v in values:
                parts.extend((
                    b'--' + boundary,
                    ('Content-Disposition: form-data; name="%s"' %
                     k).encode('utf-8'),
                    b'',
                    v.encode('utf-8')))
        for key, filename, value in files:
            parts.extend((
                b'--' + boundary,
                ('Content-Disposition: form-data; name="%s"; filename="%s"' %
                 (key, filename)).encode('utf-8'),
                b'',
                value))
        parts.extend((b'--' + boundary + b'--', b''))
        body = b'\r\n'.join(parts)
        ct = b'multipart/form-data; boundary=' + boundary
        headers = {
            'Content-type': ct,
            'Content-length': str(len(body))
        }
        return Request(self.url, body, headers)
    def search(self, terms, operator=None):
        if isinstance(terms, string_types):
            terms = {'name': terms}
        rpc_proxy = ServerProxy(self.url, timeout=3.0)
        try:
            return rpc_proxy.search(terms, operator or 'and')
        finally:
            rpc_proxy('close')()
