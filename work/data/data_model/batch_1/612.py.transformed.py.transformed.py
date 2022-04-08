
__all__ = ['HTTPRangeRequestUnsupported', 'dist_from_wheel_url']
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from zipfile import BadZipfile, ZipFile
from pip._vendor.pkg_resources import Distribution
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.wheel import pkg_resources_distribution_for_wheel
class HTTPRangeRequestUnsupported(Exception):
    pass
def dist_from_wheel_url(name, url, session):
    with LazyZipOverHTTP(url, session) as wheel:
        zip_file = ZipFile(wheel)
        return pkg_resources_distribution_for_wheel(zip_file, name, wheel.name)
class LazyZipOverHTTP:
    def __init__(self, url, session, chunk_size=CONTENT_CHUNK_SIZE):
        head = session.head(url, headers=HEADERS)
        raise_for_status(head)
        assert head.status_code == 200
        self._session, self._url, self._chunk_size = session, url, chunk_size
        self._length = int(head.headers['Content-Length'])
        self._file = NamedTemporaryFile()
        self.truncate(self._length)
        self._left = []
        self._right = []
        if 'bytes' not in head.headers.get('Accept-Ranges', 'none'):
            raise HTTPRangeRequestUnsupported('range request is not supported')
        self._check_zip()
    @property
    def mode(self):
        return 'rb'
    @property
    def name(self):
        return self._file.name
    def seekable(self):
        return True
    def close(self):
        self._file.close()
    @property
    def closed(self):
        return self._file.closed
    def read(self, size=-1):
        download_size = max(size, self._chunk_size)
        start, length = self.tell(), self._length
        stop = length if size < 0 else min(start+download_size, length)
        start = max(0, stop-download_size)
        self._download(start, stop-1)
        return self._file.read(size)
    def readable(self):
        return True
    def seek(self, offset, whence=0):
        return self._file.seek(offset, whence)
    def tell(self):
        return self._file.tell()
    def truncate(self, size=None):
        return self._file.truncate(size)
    def writable(self):
        return False
    def __enter__(self):
        self._file.__enter__()
        return self
    def __exit__(self, *exc):
        return self._file.__exit__(*exc)
    @contextmanager
    def _stay(self):
        pos = self.tell()
        try:
            yield
        finally:
            self.seek(pos)
    def _check_zip(self):
        end = self._length - 1
        for start in reversed(range(0, end, self._chunk_size)):
            self._download(start, end)
            with self._stay():
                try:
                    ZipFile(self)
                except BadZipfile:
                    pass
                else:
                    break
    def _stream_response(self, start, end, base_headers=HEADERS):
        headers = base_headers.copy()
        headers['Range'] = f'bytes={start}-{end}'
        headers['Cache-Control'] = 'no-cache'
        return self._session.get(self._url, headers=headers, stream=True)
    def _merge(self, start, end, left, right):
        lslice, rslice = self._left[left:right], self._right[left:right]
        i = start = min([start]+lslice[:1])
        end = max([end]+rslice[-1:])
        for j, k in zip(lslice, rslice):
            if j > i:
                yield i, j-1
            i = k + 1
        if i <= end:
            yield i, end
        self._left[left:right], self._right[left:right] = [start], [end]
    def _download(self, start, end):
        with self._stay():
            left = bisect_left(self._right, start)
            right = bisect_right(self._left, end)
            for start, end in self._merge(start, end, left, right):
                response = self._stream_response(start, end)
                response.raise_for_status()
                self.seek(start)
                for chunk in response_chunks(response, self._chunk_size):
                    self._file.write(chunk)
