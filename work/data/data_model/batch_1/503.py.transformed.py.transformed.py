
import datetime
import math
import os
from hashlib import md5
from bson.son import SON
from bson.binary import Binary
from bson.objectid import ObjectId
from bson.py3compat import text_type, StringIO
from gridfs.errors import CorruptGridFile, FileExists, NoFile
from pymongo import ASCENDING
from pymongo.collection import Collection
from pymongo.cursor import Cursor
from pymongo.errors import (ConfigurationError,
                            DuplicateKeyError,
                            OperationFailure)
from pymongo.read_preferences import ReadPreference
try:
    _SEEK_SET = os.SEEK_SET
    _SEEK_CUR = os.SEEK_CUR
    _SEEK_END = os.SEEK_END
except AttributeError:
    _SEEK_SET = 0
    _SEEK_CUR = 1
    _SEEK_END = 2
EMPTY = b""
NEWLN = b"\n"
DEFAULT_CHUNK_SIZE = 255 * 1024
_C_INDEX = SON([("files_id", ASCENDING), ("n", ASCENDING)])
_F_INDEX = SON([("filename", ASCENDING), ("uploadDate", ASCENDING)])
def _grid_in_property(field_name, docstring, read_only=False,
                      closed_only=False):
    def getter(self):
        if closed_only and not self._closed:
            raise AttributeError("can only get %r on a closed file" %
                                 field_name)
        if field_name == 'length':
            return self._file.get(field_name, 0)
        return self._file.get(field_name, None)
    def setter(self, value):
        if self._closed:
            self._coll.files.update_one({"_id": self._file["_id"]},
                                        {"$set": {field_name: value}})
        self._file[field_name] = value
    if read_only:
        docstring += "\n\nThis attribute is read-only."
    elif closed_only:
        docstring = "%s\n\n%s" % (docstring, "This attribute is read-only and "
                                  "can only be read after :meth:`close` "
                                  "has been called.")
    if not read_only and not closed_only:
        return property(getter, setter, doc=docstring)
    return property(getter, doc=docstring)
def _grid_out_property(field_name, docstring):
    def getter(self):
        self._ensure_file()
        if field_name == 'length':
            return self._file.get(field_name, 0)
        return self._file.get(field_name, None)
    docstring += "\n\nThis attribute is read-only."
    return property(getter, doc=docstring)
class GridIn(object):
    def __init__(self, root_collection, **kwargs):
        if not isinstance(root_collection, Collection):
            raise TypeError("root_collection must be an "
                            "instance of Collection")
        if not root_collection.write_concern.acknowledged:
            raise ConfigurationError('root_collection must use '
                                     'acknowledged write_concern')
        if "content_type" in kwargs:
            kwargs["contentType"] = kwargs.pop("content_type")
        if "chunk_size" in kwargs:
            kwargs["chunkSize"] = kwargs.pop("chunk_size")
        coll = root_collection.with_options(
            read_preference=ReadPreference.PRIMARY)
        kwargs['md5'] = md5()
        kwargs["_id"] = kwargs.get("_id", ObjectId())
        kwargs["chunkSize"] = kwargs.get("chunkSize", DEFAULT_CHUNK_SIZE)
        object.__setattr__(self, "_coll", coll)
        object.__setattr__(self, "_chunks", coll.chunks)
        object.__setattr__(self, "_file", kwargs)
        object.__setattr__(self, "_buffer", StringIO())
        object.__setattr__(self, "_position", 0)
        object.__setattr__(self, "_chunk_number", 0)
        object.__setattr__(self, "_closed", False)
        object.__setattr__(self, "_ensured_index", False)
    def __create_index(self, collection, index_key, unique):
        doc = collection.find_one(projection={"_id": 1})
        if doc is None:
            try:
                index_keys =[index_spec['key'] for index_spec in collection.list_indexes()]
            except OperationFailure:
                index_keys = []
            if index_key not in index_keys:
                collection.create_index(index_key.items(), unique=unique)
    def __ensure_indexes(self):
        if not object.__getattribute__(self, "_ensured_index"):
            self.__create_index(self._coll.files, _F_INDEX, False)
            self.__create_index(self._coll.chunks, _C_INDEX, True)
            object.__setattr__(self, "_ensured_index", True)
    def abort(self):
        self._coll.chunks.delete_many({"files_id": self._file['_id']})
        self._coll.files.delete_one({"_id": self._file['_id']})
        object.__setattr__(self, "_closed", True)
    @property
    def closed(self):
        return self._closed
    _id = _grid_in_property("_id", "The ``'_id'`` value for this file.",
                            read_only=True)
    filename = _grid_in_property("filename", "Name of this file.")
    name = _grid_in_property("filename", "Alias for `filename`.")
    content_type = _grid_in_property("contentType", "Mime-type for this file.")
    length = _grid_in_property("length", "Length (in bytes) of this file.",
                               closed_only=True)
    chunk_size = _grid_in_property("chunkSize", "Chunk size for this file.",
                                   read_only=True)
    upload_date = _grid_in_property("uploadDate",
                                    "Date that this file was uploaded.",
                                    closed_only=True)
    md5 = _grid_in_property("md5", "MD5 of the contents of this file "
                            "(generated on the server).",
                            closed_only=True)
    def __getattr__(self, name):
        if name in self._file:
            return self._file[name]
        raise AttributeError("GridIn object has no attribute '%s'" % name)
    def __setattr__(self, name, value):
        if name in self.__dict__ or name in self.__class__.__dict__:
            object.__setattr__(self, name, value)
        else:
            self._file[name] = value
            if self._closed:
                self._coll.files.update_one({"_id": self._file["_id"]},
                                            {"$set": {name: value}})
    def __flush_data(self, data):
        self.__ensure_indexes()
        self._file['md5'].update(data)
        if not data:
            return
        assert(len(data) <= self.chunk_size)
        chunk = {"files_id": self._file["_id"],
                 "n": self._chunk_number,
                 "data": Binary(data)}
        try:
            self._chunks.insert_one(chunk)
        except DuplicateKeyError:
            self._raise_file_exists(self._file['_id'])
        self._chunk_number += 1
        self._position += len(data)
    def __flush_buffer(self):
        self.__flush_data(self._buffer.getvalue())
        self._buffer.close()
        self._buffer = StringIO()
    def __flush(self):
        try:
            self.__flush_buffer()
            self._file['md5'] = self._file["md5"].hexdigest()
            self._file["length"] = self._position
            self._file["uploadDate"] = datetime.datetime.utcnow()
            return self._coll.files.insert_one(self._file)
        except DuplicateKeyError:
            self._raise_file_exists(self._id)
    def _raise_file_exists(self, file_id):
        raise FileExists("file with _id %r already exists" % file_id)
    def close(self):
        if not self._closed:
            self.__flush()
            object.__setattr__(self, "_closed", True)
    def write(self, data):
        if self._closed:
            raise ValueError("cannot write to a closed file")
        try:
            read = data.read
        except AttributeError:
            if not isinstance(data, (text_type, bytes)):
                raise TypeError("can only write strings or file-like objects")
            if isinstance(data, text_type):
                try:
                    data = data.encode(self.encoding)
                except AttributeError:
                    raise TypeError("must specify an encoding for file in "
                                    "order to write %s" % (text_type.__name__,))
            read = StringIO(data).read
        if self._buffer.tell() > 0:
            space = self.chunk_size - self._buffer.tell()
            if space:
                try:
                    to_write = read(space)
                except:
                    self.abort()
                    raise
                self._buffer.write(to_write)
                if len(to_write) < space:
                    return
            self.__flush_buffer()
        to_write = read(self.chunk_size)
        while to_write and len(to_write) == self.chunk_size:
            self.__flush_data(to_write)
            to_write = read(self.chunk_size)
        self._buffer.write(to_write)
    def writelines(self, sequence):
        for line in sequence:
            self.write(line)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
class GridOut(object):
    def __init__(self, root_collection, file_id=None, file_document=None):
        if not isinstance(root_collection, Collection):
            raise TypeError("root_collection must be an "
                            "instance of Collection")
        self.__chunks = root_collection.chunks
        self.__files = root_collection.files
        self.__file_id = file_id
        self.__buffer = EMPTY
        self.__position = 0
        self._file = file_document
    _id = _grid_out_property("_id", "The ``'_id'`` value for this file.")
    filename = _grid_out_property("filename", "Name of this file.")
    name = _grid_out_property("filename", "Alias for `filename`.")
    content_type = _grid_out_property("contentType", "Mime-type for this file.")
    length = _grid_out_property("length", "Length (in bytes) of this file.")
    chunk_size = _grid_out_property("chunkSize", "Chunk size for this file.")
    upload_date = _grid_out_property("uploadDate",
                                     "Date that this file was first uploaded.")
    aliases = _grid_out_property("aliases", "List of aliases for this file.")
    metadata = _grid_out_property("metadata", "Metadata attached to this file.")
    md5 = _grid_out_property("md5", "MD5 of the contents of this file "
                             "(generated on the server).")
    def _ensure_file(self):
        if not self._file:
            self._file = self.__files.find_one({"_id": self.__file_id})
            if not self._file:
                raise NoFile("no file in gridfs collection %r with _id %r" %
                             (self.__files, self.__file_id))
    def __getattr__(self, name):
        self._ensure_file()
        if name in self._file:
            return self._file[name]
        raise AttributeError("GridOut object has no attribute '%s'" % name)
    def readchunk(self):
        received = len(self.__buffer)
        chunk_data = EMPTY
        chunk_size = int(self.chunk_size)
        if received > 0:
            chunk_data = self.__buffer
        elif self.__position < int(self.length):
            chunk_number = int((received + self.__position) / chunk_size)
            chunk = self.__chunks.find_one({"files_id": self._id,
                                            "n": chunk_number})
            if not chunk:
            chunk_data = chunk["data"][self.__position % chunk_size:]
            if not chunk_data:
                raise CorruptGridFile("truncated chunk")
        self.__position += len(chunk_data)
        self.__buffer = EMPTY
        return chunk_data
    def read(self, size=-1):
        self._ensure_file()
        if size == 0:
            return EMPTY
        remainder = int(self.length) - self.__position
        if size < 0 or size > remainder:
            size = remainder
        received = 0
        data = StringIO()
        while received < size:
            chunk_data = self.readchunk()
            received += len(chunk_data)
            data.write(chunk_data)
        max_chunk_n = math.ceil(self.length / float(self.chunk_size))
        chunk = self.__chunks.find_one({"files_id": self._id,
                                        "n": {"$gte": max_chunk_n}})
        if chunk is not None and len(chunk['data']):
            raise CorruptGridFile(
                "Extra chunk found: expected %i chunks but found "
                "chunk with n=%i" % (max_chunk_n, chunk['n']))
        self.__position -= received - size
        data.seek(size)
        self.__buffer = data.read()
        data.seek(0)
        return data.read(size)
    def readline(self, size=-1):
        if size == 0:
            return b''
        remainder = int(self.length) - self.__position
        if size < 0 or size > remainder:
            size = remainder
        received = 0
        data = StringIO()
        while received < size:
            chunk_data = self.readchunk()
            pos = chunk_data.find(NEWLN, 0, size)
            if pos != -1:
                size = received + pos + 1
            received += len(chunk_data)
            data.write(chunk_data)
            if pos != -1:
                break
        self.__position -= received - size
        data.seek(size)
        self.__buffer = data.read()
        data.seek(0)
        return data.read(size)
    def tell(self):
        return self.__position
    def seek(self, pos, whence=_SEEK_SET):
        if whence == _SEEK_SET:
            new_pos = pos
        elif whence == _SEEK_CUR:
            new_pos = self.__position + pos
        elif whence == _SEEK_END:
            new_pos = int(self.length) + pos
        else:
            raise IOError(22, "Invalid value for `whence`")
        if new_pos < 0:
            raise IOError(22, "Invalid value for `pos` - must be positive")
        self.__position = new_pos
        self.__buffer = EMPTY
    def __iter__(self):
        return GridOutIterator(self, self.__chunks)
    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
class GridOutIterator(object):
    def __init__(self, grid_out, chunks):
        self.__id = grid_out._id
        self.__chunks = chunks
        self.__current_chunk = 0
        self.__max_chunk = math.ceil(float(grid_out.length) /
                                     grid_out.chunk_size)
    def __iter__(self):
        return self
    def next(self):
        if self.__current_chunk >= self.__max_chunk:
            raise StopIteration
        chunk = self.__chunks.find_one({"files_id": self.__id,
                                        "n": self.__current_chunk})
        if not chunk:
        self.__current_chunk += 1
        return bytes(chunk["data"])
    __next__ = next
class GridOutCursor(Cursor):
    def __init__(self, collection, filter=None, skip=0, limit=0,
                 no_cursor_timeout=False, sort=None, batch_size=0):
        self.__root_collection = collection
        super(GridOutCursor, self).__init__(
            collection.files, filter, skip=skip, limit=limit,
            no_cursor_timeout=no_cursor_timeout, sort=sort,
            batch_size=batch_size)
    def next(self):
        next_file = super(GridOutCursor, self).next()
        return GridOut(self.__root_collection, file_document=next_file)
    __next__ = next
    def add_option(self, *args, **kwargs):
        raise NotImplementedError("Method does not exist for GridOutCursor")
    def remove_option(self, *args, **kwargs):
        raise NotImplementedError("Method does not exist for GridOutCursor")
    def _clone_base(self):
        return GridOutCursor(self.__root_collection)
