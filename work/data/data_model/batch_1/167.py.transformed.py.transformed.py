
import collections
import datetime
import warnings
from bson.code import Code
from bson.objectid import ObjectId
from bson.py3compat import (_unicode,
                            integer_types,
                            string_type)
from bson.raw_bson import RawBSONDocument
from bson.codec_options import CodecOptions
from bson.son import SON
from pymongo import (common,
                     helpers,
                     message)
from pymongo.bulk import BulkOperationBuilder, _Bulk
from pymongo.command_cursor import CommandCursor
from pymongo.collation import validate_collation_or_none
from pymongo.cursor import Cursor
from pymongo.errors import ConfigurationError, InvalidName, OperationFailure
from pymongo.helpers import _check_write_command_response
from pymongo.helpers import _UNICODE_REPLACE_CODEC_OPTIONS
from pymongo.operations import IndexModel
from pymongo.read_concern import DEFAULT_READ_CONCERN
from pymongo.read_preferences import ReadPreference
from pymongo.results import (BulkWriteResult,
                             DeleteResult,
                             InsertOneResult,
                             InsertManyResult,
                             UpdateResult)
from pymongo.write_concern import WriteConcern
try:
    from collections import OrderedDict
    _ORDERED_TYPES = (SON, OrderedDict)
except ImportError:
    _ORDERED_TYPES = (SON,)
_NO_OBJ_ERROR = "No matching object found"
_UJOIN = u"%s.%s"
class ReturnDocument(object):
    BEFORE = False
    AFTER = True
class Collection(common.BaseObject):
    def __init__(self, database, name, create=False, codec_options=None,
                 read_preference=None, write_concern=None, read_concern=None,
                 **kwargs):
        super(Collection, self).__init__(
            codec_options or database.codec_options,
            read_preference or database.read_preference,
            write_concern or database.write_concern,
            read_concern or database.read_concern)
        if not isinstance(name, string_type):
            raise TypeError("name must be an instance "
                            "of %s" % (string_type.__name__,))
        if not name or ".." in name:
            raise InvalidName("collection names cannot be empty")
        if "$" in name and not (name.startswith("oplog.$main") or
                                name.startswith("$cmd")):
            raise InvalidName("collection names must not "
                              "contain '$': %r" % name)
        if name[0] == "." or name[-1] == ".":
            raise InvalidName("collection names must not start "
                              "or end with '.': %r" % name)
        if "\x00" in name:
            raise InvalidName("collection names must not contain the "
                              "null character")
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        self.__database = database
        self.__name = _unicode(name)
        self.__full_name = _UJOIN % (self.__database.name, self.__name)
        if create or kwargs or collation:
            self.__create(kwargs, collation)
        self.__write_response_codec_options = self.codec_options._replace(
            unicode_decode_error_handler='replace',
            document_class=dict)
    def _socket_for_reads(self):
        return self.__database.client._socket_for_reads(self.read_preference)
    def _socket_for_primary_reads(self):
        return self.__database.client._socket_for_reads(ReadPreference.PRIMARY)
    def _socket_for_writes(self):
        return self.__database.client._socket_for_writes()
    def _command(self, sock_info, command, slave_ok=False,
                 read_preference=None,
                 codec_options=None, check=True, allowable_errors=None,
                 read_concern=DEFAULT_READ_CONCERN,
                 write_concern=None,
                 parse_write_concern_error=False,
                 collation=None):
        return sock_info.command(
            self.__database.name,
            command,
            slave_ok,
            read_preference or self.read_preference,
            codec_options or self.codec_options,
            check,
            allowable_errors,
            read_concern=read_concern,
            write_concern=write_concern,
            parse_write_concern_error=parse_write_concern_error,
            collation=collation)
    def __create(self, options, collation):
        cmd = SON([("create", self.__name)])
        if options:
            if "size" in options:
                options["size"] = float(options["size"])
            cmd.update(options)
        with self._socket_for_writes() as sock_info:
            self._command(
                sock_info, cmd, read_preference=ReadPreference.PRIMARY,
                write_concern=self.write_concern,
                parse_write_concern_error=True,
                collation=collation)
    def __getattr__(self, name):
        if name.startswith('_'):
            full_name = _UJOIN % (self.__name, name)
            raise AttributeError(
                "Collection has no attribute %r. To access the %s"
                " collection, use database['%s']." % (
                    name, full_name, full_name))
        return self.__getitem__(name)
    def __getitem__(self, name):
        return Collection(self.__database, _UJOIN % (self.__name, name))
    def __repr__(self):
        return "Collection(%r, %r)" % (self.__database, self.__name)
    def __eq__(self, other):
        if isinstance(other, Collection):
            return (self.__database == other.database and
                    self.__name == other.name)
        return NotImplemented
    def __ne__(self, other):
        return not self == other
    @property
    def full_name(self):
        return self.__full_name
    @property
    def name(self):
        return self.__name
    @property
    def database(self):
        return self.__database
    def with_options(
            self, codec_options=None, read_preference=None,
            write_concern=None, read_concern=None):
        return Collection(self.__database,
                          self.__name,
                          False,
                          codec_options or self.codec_options,
                          read_preference or self.read_preference,
                          write_concern or self.write_concern,
                          read_concern or self.read_concern)
    def initialize_unordered_bulk_op(self, bypass_document_validation=False):
        warnings.warn("initialize_unordered_bulk_op is deprecated",
                      DeprecationWarning, stacklevel=2)
        return BulkOperationBuilder(self, False, bypass_document_validation)
    def initialize_ordered_bulk_op(self, bypass_document_validation=False):
        warnings.warn("initialize_ordered_bulk_op is deprecated",
                      DeprecationWarning, stacklevel=2)
        return BulkOperationBuilder(self, True, bypass_document_validation)
    def bulk_write(self, requests, ordered=True,
                   bypass_document_validation=False):
        if not isinstance(requests, list):
            raise TypeError("requests must be a list")
        blk = _Bulk(self, ordered, bypass_document_validation)
        for request in requests:
            try:
                request._add_to_bulk(blk)
            except AttributeError:
                raise TypeError("%r is not a valid request" % (request,))
        bulk_api_result = blk.execute(self.write_concern.document)
        if bulk_api_result is not None:
            return BulkWriteResult(bulk_api_result, True)
        return BulkWriteResult({}, False)
    def _legacy_write(self, sock_info, name, cmd, acknowledged, op_id,
                      bypass_doc_val, func, *args):
        if (bypass_doc_val and not acknowledged and
                    sock_info.max_wire_version >= 4):
            raise OperationFailure("Cannot set bypass_document_validation with"
                                   " unacknowledged write concern")
        listeners = self.database.client._event_listeners
        publish = listeners.enabled_for_commands
        if publish:
            start = datetime.datetime.now()
        rqst_id, msg, max_size = func(*args)
        if publish:
            duration = datetime.datetime.now() - start
            listeners.publish_command_start(
                cmd, self.__database.name, rqst_id, sock_info.address, op_id)
            start = datetime.datetime.now()
        try:
            result = sock_info.legacy_write(
                rqst_id, msg, max_size, acknowledged)
        except Exception as exc:
            if publish:
                dur = (datetime.datetime.now() - start) + duration
                if isinstance(exc, OperationFailure):
                    details = exc.details
                    if details.get("ok") and "n" in details:
                        reply = message._convert_write_result(
                            name, cmd, details)
                        listeners.publish_command_success(
                            dur, reply, name, rqst_id, sock_info.address, op_id)
                        raise
                else:
                    details = message._convert_exception(exc)
                listeners.publish_command_failure(
                    dur, details, name, rqst_id, sock_info.address, op_id)
            raise
        if publish:
            if result is not None:
                reply = message._convert_write_result(name, cmd, result)
            else:
                reply = {'ok': 1}
            duration = (datetime.datetime.now() - start) + duration
            listeners.publish_command_success(
                duration, reply, name, rqst_id, sock_info.address, op_id)
        return result
    def _insert_one(
            self, sock_info, doc, ordered,
            check_keys, manipulate, write_concern, op_id, bypass_doc_val):
        if manipulate:
            doc = self.__database._apply_incoming_manipulators(doc, self)
            if not isinstance(doc, RawBSONDocument) and '_id' not in doc:
                doc['_id'] = ObjectId()
            doc = self.__database._apply_incoming_copying_manipulators(doc,
                                                                       self)
        concern = (write_concern or self.write_concern).document
        acknowledged = concern.get("w") != 0
        command = SON([('insert', self.name),
                       ('ordered', ordered),
                       ('documents', [doc])])
        if concern:
            command['writeConcern'] = concern
        if sock_info.max_wire_version > 1 and acknowledged:
            if bypass_doc_val and sock_info.max_wire_version >= 4:
                command['bypassDocumentValidation'] = True
            result = sock_info.command(
                self.__database.name,
                command,
                codec_options=self.__write_response_codec_options,
                check_keys=check_keys)
            _check_write_command_response([(0, result)])
        else:
            self._legacy_write(
                sock_info, 'insert', command, acknowledged, op_id,
                bypass_doc_val, message.insert, self.__full_name, [doc],
                check_keys, acknowledged, concern, False,
                self.__write_response_codec_options)
        if not isinstance(doc, RawBSONDocument):
            return doc.get('_id')
    def _insert(self, sock_info, docs, ordered=True, check_keys=True,
                manipulate=False, write_concern=None, op_id=None,
                bypass_doc_val=False):
        if isinstance(docs, collections.Mapping):
            return self._insert_one(
                sock_info, docs, ordered,
                check_keys, manipulate, write_concern, op_id, bypass_doc_val)
        ids = []
        if manipulate:
            def gen():
                _db = self.__database
                for doc in docs:
                    doc = _db._apply_incoming_manipulators(doc, self)
                    if not (isinstance(doc, RawBSONDocument) or '_id' in doc):
                        doc['_id'] = ObjectId()
                    doc = _db._apply_incoming_copying_manipulators(doc, self)
                    ids.append(doc['_id'])
                    yield doc
        else:
            def gen():
                for doc in docs:
                    if not isinstance(doc, RawBSONDocument):
                        ids.append(doc.get('_id'))
                    yield doc
        concern = (write_concern or self.write_concern).document
        acknowledged = concern.get("w") != 0
        command = SON([('insert', self.name),
                       ('ordered', ordered)])
        if concern:
            command['writeConcern'] = concern
        if op_id is None:
            op_id = message._randint()
        if bypass_doc_val and sock_info.max_wire_version >= 4:
            command['bypassDocumentValidation'] = True
        bwc = message._BulkWriteContext(
            self.database.name, command, sock_info, op_id,
            self.database.client._event_listeners)
        if sock_info.max_wire_version > 1 and acknowledged:
            results = message._do_batched_write_command(
                self.database.name + ".$cmd", message._INSERT, command,
                gen(), check_keys, self.__write_response_codec_options, bwc)
            _check_write_command_response(results)
        else:
            message._do_batched_insert(self.__full_name, gen(), check_keys,
                                       acknowledged, concern, not ordered,
                                       self.__write_response_codec_options, bwc)
        return ids
    def insert_one(self, document, bypass_document_validation=False):
        common.validate_is_document_type("document", document)
        if not (isinstance(document, RawBSONDocument) or "_id" in document):
            document["_id"] = ObjectId()
        with self._socket_for_writes() as sock_info:
            return InsertOneResult(
                self._insert(sock_info, document,
                             bypass_doc_val=bypass_document_validation),
                self.write_concern.acknowledged)
    def insert_many(self, documents, ordered=True,
                    bypass_document_validation=False):
        if not isinstance(documents, collections.Iterable) or not documents:
            raise TypeError("documents must be a non-empty list")
        inserted_ids = []
        def gen():
            for document in documents:
                common.validate_is_document_type("document", document)
                if not isinstance(document, RawBSONDocument):
                    if "_id" not in document:
                        document["_id"] = ObjectId()
                    inserted_ids.append(document["_id"])
                yield (message._INSERT, document)
        blk = _Bulk(self, ordered, bypass_document_validation)
        blk.ops = [doc for doc in gen()]
        blk.execute(self.write_concern.document)
        return InsertManyResult(inserted_ids, self.write_concern.acknowledged)
    def _update(self, sock_info, criteria, document, upsert=False,
                check_keys=True, multi=False, manipulate=False,
                write_concern=None, op_id=None, ordered=True,
                bypass_doc_val=False, collation=None):
        common.validate_boolean("upsert", upsert)
        if manipulate:
            document = self.__database._fix_incoming(document, self)
        collation = validate_collation_or_none(collation)
        concern = (write_concern or self.write_concern).document
        acknowledged = concern.get("w") != 0
        update_doc = SON([('q', criteria),
                          ('u', document),
                          ('multi', multi),
                          ('upsert', upsert)])
        if collation is not None:
            if sock_info.max_wire_version < 5:
                raise ConfigurationError(
                    'Must be connected to MongoDB 3.4+ to use collations.')
            elif not acknowledged:
                raise ConfigurationError(
                    'Collation is unsupported for unacknowledged writes.')
            else:
                update_doc['collation'] = collation
        command = SON([('update', self.name),
                       ('ordered', ordered),
                       ('updates', [update_doc])])
        if concern:
            command['writeConcern'] = concern
        if sock_info.max_wire_version > 1 and acknowledged:
            if bypass_doc_val and sock_info.max_wire_version >= 4:
                command['bypassDocumentValidation'] = True
            result = sock_info.command(
                self.__database.name,
                command,
                codec_options=self.__write_response_codec_options).copy()
            _check_write_command_response([(0, result)])
            if result.get('n') and 'upserted' not in result:
                result['updatedExisting'] = True
            else:
                result['updatedExisting'] = False
                if 'upserted' in result:
                    result['upserted'] = result['upserted'][0]['_id']
            return result
        else:
            return self._legacy_write(
                sock_info, 'update', command, acknowledged, op_id,
                bypass_doc_val, message.update, self.__full_name, upsert,
                multi, criteria, document, acknowledged, concern, check_keys,
                self.__write_response_codec_options)
    def replace_one(self, filter, replacement, upsert=False,
                    bypass_document_validation=False, collation=None):
        common.validate_is_mapping("filter", filter)
        common.validate_ok_for_replace(replacement)
        with self._socket_for_writes() as sock_info:
            result = self._update(sock_info, filter, replacement, upsert,
                                  bypass_doc_val=bypass_document_validation,
                                  collation=collation)
        return UpdateResult(result, self.write_concern.acknowledged)
    def update_one(self, filter, update, upsert=False,
                   bypass_document_validation=False,
                   collation=None):
        common.validate_is_mapping("filter", filter)
        common.validate_ok_for_update(update)
        with self._socket_for_writes() as sock_info:
            result = self._update(sock_info, filter, update, upsert,
                                  check_keys=False,
                                  bypass_doc_val=bypass_document_validation,
                                  collation=collation)
        return UpdateResult(result, self.write_concern.acknowledged)
    def update_many(self, filter, update, upsert=False,
                    bypass_document_validation=False, collation=None):
        common.validate_is_mapping("filter", filter)
        common.validate_ok_for_update(update)
        with self._socket_for_writes() as sock_info:
            result = self._update(sock_info, filter, update, upsert,
                                  check_keys=False, multi=True,
                                  bypass_doc_val=bypass_document_validation,
                                  collation=collation)
        return UpdateResult(result, self.write_concern.acknowledged)
    def drop(self):
        self.__database.drop_collection(self.__name)
    def _delete(
            self, sock_info, criteria, multi,
            write_concern=None, op_id=None, ordered=True,
            collation=None):
        common.validate_is_mapping("filter", criteria)
        concern = (write_concern or self.write_concern).document
        acknowledged = concern.get("w") != 0
        delete_doc = SON([('q', criteria),
                          ('limit', int(not multi))])
        collation = validate_collation_or_none(collation)
        if collation is not None:
            if sock_info.max_wire_version < 5:
                raise ConfigurationError(
                    'Must be connected to MongoDB 3.4+ to use collations.')
            elif not acknowledged:
                raise ConfigurationError(
                    'Collation is unsupported for unacknowledged writes.')
            else:
                delete_doc['collation'] = collation
        command = SON([('delete', self.name),
                       ('ordered', ordered),
                       ('deletes', [delete_doc])])
        if concern:
            command['writeConcern'] = concern
        if sock_info.max_wire_version > 1 and acknowledged:
            result = sock_info.command(
                self.__database.name,
                command,
                codec_options=self.__write_response_codec_options)
            _check_write_command_response([(0, result)])
            return result
        else:
            return self._legacy_write(
                sock_info, 'delete', command, acknowledged, op_id,
                False, message.delete, self.__full_name, criteria,
                acknowledged, concern, self.__write_response_codec_options,
                int(not multi))
    def delete_one(self, filter, collation=None):
        with self._socket_for_writes() as sock_info:
            return DeleteResult(self._delete(sock_info, filter, False,
                                             collation=collation),
                                self.write_concern.acknowledged)
    def delete_many(self, filter, collation=None):
        with self._socket_for_writes() as sock_info:
            return DeleteResult(self._delete(sock_info, filter, True,
                                             collation=collation),
                                self.write_concern.acknowledged)
    def find_one(self, filter=None, *args, **kwargs):
        if (filter is not None and not
                isinstance(filter, collections.Mapping)):
            filter = {"_id": filter}
        cursor = self.find(filter, *args, **kwargs)
        for result in cursor.limit(-1):
            return result
        return None
    def find(self, *args, **kwargs):
        return Cursor(self, *args, **kwargs)
    def parallel_scan(self, num_cursors, **kwargs):
        cmd = SON([('parallelCollectionScan', self.__name),
                   ('numCursors', num_cursors)])
        cmd.update(kwargs)
        with self._socket_for_reads() as (sock_info, slave_ok):
            result = self._command(sock_info, cmd, slave_ok,
                                   read_concern=self.read_concern)
        return [CommandCursor(self, cursor['cursor'], sock_info.address)
                for cursor in result['cursors']]
    def _count(self, cmd, collation=None):
        with self._socket_for_reads() as (sock_info, slave_ok):
            res = self._command(
                sock_info, cmd, slave_ok,
                allowable_errors=["ns missing"],
                codec_options=self.__write_response_codec_options,
                read_concern=self.read_concern,
                collation=collation)
        if res.get("errmsg", "") == "ns missing":
            return 0
        return int(res["n"])
    def count(self, filter=None, **kwargs):
        cmd = SON([("count", self.__name)])
        if filter is not None:
            if "query" in kwargs:
                raise ConfigurationError("can't pass both filter and query")
            kwargs["query"] = filter
        if "hint" in kwargs and not isinstance(kwargs["hint"], string_type):
            kwargs["hint"] = helpers._index_document(kwargs["hint"])
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd.update(kwargs)
        return self._count(cmd, collation)
    def create_indexes(self, indexes):
        if not isinstance(indexes, list):
            raise TypeError("indexes must be a list")
        names = []
        def gen_indexes():
            for index in indexes:
                if not isinstance(index, IndexModel):
                    raise TypeError("%r is not an instance of "
                                    "pymongo.operations.IndexModel" % (index,))
                document = index.document
                names.append(document["name"])
                yield document
        cmd = SON([('createIndexes', self.name),
                   ('indexes', list(gen_indexes()))])
        with self._socket_for_writes() as sock_info:
            self._command(
                sock_info, cmd, read_preference=ReadPreference.PRIMARY,
                codec_options=_UNICODE_REPLACE_CODEC_OPTIONS,
                write_concern=self.write_concern,
                parse_write_concern_error=True)
        return names
    def __create_index(self, keys, index_options):
        index_doc = helpers._index_document(keys)
        index = {"key": index_doc}
        collation = validate_collation_or_none(
            index_options.pop('collation', None))
        index.update(index_options)
        with self._socket_for_writes() as sock_info:
            if collation is not None:
                if sock_info.max_wire_version < 5:
                    raise ConfigurationError(
                        'Must be connected to MongoDB 3.4+ to use collations.')
                else:
                    index['collation'] = collation
            cmd = SON([('createIndexes', self.name), ('indexes', [index])])
            try:
                self._command(
                    sock_info, cmd, read_preference=ReadPreference.PRIMARY,
                    codec_options=_UNICODE_REPLACE_CODEC_OPTIONS,
                    write_concern=self.write_concern,
                    parse_write_concern_error=True)
            except OperationFailure as exc:
                if exc.code in common.COMMAND_NOT_FOUND_CODES:
                    index["ns"] = self.__full_name
                    wcn = (self.write_concern if
                           self.write_concern.acknowledged else WriteConcern())
                    self.__database.system.indexes._insert(
                        sock_info, index, True, False, False, wcn)
                else:
                    raise
    def create_index(self, keys, **kwargs):
        keys = helpers._index_list(keys)
        name = kwargs.setdefault("name", helpers._gen_index_name(keys))
        self.__create_index(keys, kwargs)
        return name
    def ensure_index(self, key_or_list, cache_for=300, **kwargs):
        warnings.warn("ensure_index is deprecated. Use create_index instead.",
                      DeprecationWarning, stacklevel=2)
        if not (isinstance(cache_for, integer_types) or
                isinstance(cache_for, float)):
            raise TypeError("cache_for must be an integer or float.")
        if "drop_dups" in kwargs:
            kwargs["dropDups"] = kwargs.pop("drop_dups")
        if "bucket_size" in kwargs:
            kwargs["bucketSize"] = kwargs.pop("bucket_size")
        keys = helpers._index_list(key_or_list)
        name = kwargs.setdefault("name", helpers._gen_index_name(keys))
        if not self.__database.client._cached(self.__database.name,
                                              self.__name, name):
            self.__create_index(keys, kwargs)
            self.__database.client._cache_index(self.__database.name,
                                                self.__name, name, cache_for)
            return name
        return None
    def drop_indexes(self):
        self.__database.client._purge_index(self.__database.name, self.__name)
        self.drop_index("*")
    def drop_index(self, index_or_name):
        name = index_or_name
        if isinstance(index_or_name, list):
            name = helpers._gen_index_name(index_or_name)
        if not isinstance(name, string_type):
            raise TypeError("index_or_name must be an index name or list")
        self.__database.client._purge_index(
            self.__database.name, self.__name, name)
        cmd = SON([("dropIndexes", self.__name), ("index", name)])
        with self._socket_for_writes() as sock_info:
            self._command(sock_info,
                          cmd,
                          read_preference=ReadPreference.PRIMARY,
                          allowable_errors=["ns not found"],
                          write_concern=self.write_concern,
                          parse_write_concern_error=True)
    def reindex(self):
        cmd = SON([("reIndex", self.__name)])
        with self._socket_for_writes() as sock_info:
            return self._command(
                sock_info, cmd, read_preference=ReadPreference.PRIMARY,
                parse_write_concern_error=True)
    def list_indexes(self):
        codec_options = CodecOptions(SON)
        coll = self.with_options(codec_options)
        with self._socket_for_primary_reads() as (sock_info, slave_ok):
            cmd = SON([("listIndexes", self.__name), ("cursor", {})])
            if sock_info.max_wire_version > 2:
                try:
                    cursor = self._command(sock_info, cmd, slave_ok,
                                           ReadPreference.PRIMARY,
                                           codec_options)["cursor"]
                except OperationFailure as exc:
                    if exc.code != 26:
                        raise
                    cursor = {'id': 0, 'firstBatch': []}
                return CommandCursor(coll, cursor, sock_info.address)
            else:
                namespace = _UJOIN % (self.__database.name, "system.indexes")
                res = helpers._first_batch(
                    sock_info, self.__database.name, "system.indexes",
                    {"ns": self.__full_name}, 0, slave_ok, codec_options,
                    ReadPreference.PRIMARY, cmd,
                    self.database.client._event_listeners)
                data = res["data"]
                cursor = {
                    "id": res["cursor_id"],
                    "firstBatch": data,
                    "ns": namespace,
                }
                return CommandCursor(
                    coll, cursor, sock_info.address, len(data))
    def index_information(self):
        cursor = self.list_indexes()
        info = {}
        for index in cursor:
            index["key"] = index["key"].items()
            index = dict(index)
            info[index.pop("name")] = index
        return info
    def options(self):
        with self._socket_for_primary_reads() as (sock_info, slave_ok):
            if sock_info.max_wire_version > 2:
                criteria = {"name": self.__name}
            else:
                criteria = {"name": self.__full_name}
            cursor = self.__database._list_collections(sock_info,
                                                       slave_ok,
                                                       criteria)
        result = None
        for doc in cursor:
            result = doc
            break
        if not result:
            return {}
        options = result.get("options", {})
        if "create" in options:
            del options["create"]
        return options
    def aggregate(self, pipeline, **kwargs):
        if not isinstance(pipeline, list):
            raise TypeError("pipeline must be a list")
        if "explain" in kwargs:
            raise ConfigurationError("The explain option is not supported. "
                                     "Use Database.command instead.")
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd = SON([("aggregate", self.__name),
                   ("pipeline", pipeline)])
        batch_size = common.validate_positive_integer_or_none(
            "batchSize", kwargs.pop("batchSize", None))
        use_cursor = common.validate_boolean(
            "useCursor", kwargs.pop("useCursor", True))
        with self._socket_for_reads() as (sock_info, slave_ok):
            if sock_info.max_wire_version > 0:
                if use_cursor:
                    if "cursor" not in kwargs:
                        kwargs["cursor"] = {}
                    if batch_size is not None:
                        kwargs["cursor"]["batchSize"] = batch_size
            dollar_out = pipeline and '$out' in pipeline[-1]
            if (sock_info.max_wire_version >= 5 and dollar_out and
                    self.write_concern):
                cmd['writeConcern'] = self.write_concern.document
            cmd.update(kwargs)
            if sock_info.max_wire_version >= 4 and 'readConcern' not in cmd:
                if dollar_out:
                    result = self._command(sock_info, cmd, slave_ok,
                                           parse_write_concern_error=True,
                                           collation=collation)
                else:
                    result = self._command(sock_info, cmd, slave_ok,
                                           read_concern=self.read_concern,
                                           collation=collation)
            else:
                result = self._command(sock_info, cmd, slave_ok,
                                       parse_write_concern_error=dollar_out,
                                       collation=collation)
            if "cursor" in result:
                cursor = result["cursor"]
            else:
                cursor = {
                    "id": 0,
                    "firstBatch": result["result"],
                    "ns": self.full_name,
                }
            return CommandCursor(
                self, cursor, sock_info.address).batch_size(batch_size or 0)
    def group(self, key, condition, initial, reduce, finalize=None, **kwargs):
        warnings.warn("The group method is deprecated and will be removed in "
                      "PyMongo 4.0. Use the aggregate method with the $group "
                      "stage or the map_reduce method instead.",
                      DeprecationWarning, stacklevel=2)
        group = {}
        if isinstance(key, string_type):
            group["$keyf"] = Code(key)
        elif key is not None:
            group = {"key": helpers._fields_list_to_dict(key, "key")}
        group["ns"] = self.__name
        group["$reduce"] = Code(reduce)
        group["cond"] = condition
        group["initial"] = initial
        if finalize is not None:
            group["finalize"] = Code(finalize)
        cmd = SON([("group", group)])
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd.update(kwargs)
        with self._socket_for_reads() as (sock_info, slave_ok):
            return self._command(sock_info, cmd, slave_ok,
                                 collation=collation)["retval"]
    def rename(self, new_name, **kwargs):
        if not isinstance(new_name, string_type):
            raise TypeError("new_name must be an "
                            "instance of %s" % (string_type.__name__,))
        if not new_name or ".." in new_name:
            raise InvalidName("collection names cannot be empty")
        if new_name[0] == "." or new_name[-1] == ".":
            raise InvalidName("collecion names must not start or end with '.'")
        if "$" in new_name and not new_name.startswith("oplog.$main"):
            raise InvalidName("collection names must not contain '$'")
        new_name = "%s.%s" % (self.__database.name, new_name)
        cmd = SON([("renameCollection", self.__full_name), ("to", new_name)])
        with self._socket_for_writes() as sock_info:
            if sock_info.max_wire_version >= 5 and self.write_concern:
                cmd['writeConcern'] = self.write_concern.document
            cmd.update(kwargs)
            sock_info.command('admin', cmd, parse_write_concern_error=True)
    def distinct(self, key, filter=None, **kwargs):
        if not isinstance(key, string_type):
            raise TypeError("key must be an "
                            "instance of %s" % (string_type.__name__,))
        cmd = SON([("distinct", self.__name),
                   ("key", key)])
        if filter is not None:
            if "query" in kwargs:
                raise ConfigurationError("can't pass both filter and query")
            kwargs["query"] = filter
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd.update(kwargs)
        with self._socket_for_reads() as (sock_info, slave_ok):
            return self._command(sock_info, cmd, slave_ok,
                                 read_concern=self.read_concern,
                                 collation=collation)["values"]
    def map_reduce(self, map, reduce, out, full_response=False, **kwargs):
        if not isinstance(out, (string_type, collections.Mapping)):
            raise TypeError("'out' must be an instance of "
                            "%s or a mapping" % (string_type.__name__,))
        cmd = SON([("mapreduce", self.__name),
                   ("map", map),
                   ("reduce", reduce),
                   ("out", out)])
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd.update(kwargs)
        inline = 'inline' in cmd['out']
        with self._socket_for_primary_reads() as (sock_info, slave_ok):
            if (sock_info.max_wire_version >= 5 and self.write_concern and
                    not inline):
                cmd['writeConcern'] = self.write_concern.document
            cmd.update(kwargs)
            if (sock_info.max_wire_version >= 4 and 'readConcern' not in cmd and
                    inline):
                response = self._command(
                    sock_info, cmd, slave_ok, ReadPreference.PRIMARY,
                    read_concern=self.read_concern,
                    collation=collation)
            else:
                response = self._command(
                    sock_info, cmd, slave_ok, ReadPreference.PRIMARY,
                    parse_write_concern_error=not inline,
                    collation=collation)
        if full_response or not response.get('result'):
            return response
        elif isinstance(response['result'], dict):
            dbase = response['result']['db']
            coll = response['result']['collection']
            return self.__database.client[dbase][coll]
        else:
            return self.__database[response["result"]]
    def inline_map_reduce(self, map, reduce, full_response=False, **kwargs):
        cmd = SON([("mapreduce", self.__name),
                   ("map", map),
                   ("reduce", reduce),
                   ("out", {"inline": 1})])
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd.update(kwargs)
        with self._socket_for_reads() as (sock_info, slave_ok):
            if sock_info.max_wire_version >= 4 and 'readConcern' not in cmd:
                res = self._command(sock_info, cmd, slave_ok,
                                    read_concern=self.read_concern,
                                    collation=collation)
            else:
                res = self._command(sock_info, cmd, slave_ok,
                                    collation=collation)
        if full_response:
            return res
        else:
            return res.get("results")
    def __find_and_modify(self, filter, projection, sort, upsert=None,
                          return_document=ReturnDocument.BEFORE, **kwargs):
        common.validate_is_mapping("filter", filter)
        if not isinstance(return_document, bool):
            raise ValueError("return_document must be "
                             "ReturnDocument.BEFORE or ReturnDocument.AFTER")
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd = SON([("findAndModify", self.__name),
                   ("query", filter),
                   ("new", return_document)])
        cmd.update(kwargs)
        if projection is not None:
            cmd["fields"] = helpers._fields_list_to_dict(projection,
                                                         "projection")
        if sort is not None:
            cmd["sort"] = helpers._index_document(sort)
        if upsert is not None:
            common.validate_boolean("upsert", upsert)
            cmd["upsert"] = upsert
        with self._socket_for_writes() as sock_info:
            if sock_info.max_wire_version >= 4 and 'writeConcern' not in cmd:
                wc_doc = self.write_concern.document
                if wc_doc:
                    cmd['writeConcern'] = wc_doc
            out = self._command(sock_info, cmd,
                                read_preference=ReadPreference.PRIMARY,
                                allowable_errors=[_NO_OBJ_ERROR],
                                collation=collation)
            _check_write_command_response([(0, out)])
        return out.get("value")
    def find_one_and_delete(self, filter,
                            projection=None, sort=None, **kwargs):
        kwargs['remove'] = True
        return self.__find_and_modify(filter, projection, sort, **kwargs)
    def find_one_and_replace(self, filter, replacement,
                             projection=None, sort=None, upsert=False,
                             return_document=ReturnDocument.BEFORE, **kwargs):
        common.validate_ok_for_replace(replacement)
        kwargs['update'] = replacement
        return self.__find_and_modify(filter, projection,
                                      sort, upsert, return_document, **kwargs)
    def find_one_and_update(self, filter, update,
                            projection=None, sort=None, upsert=False,
                            return_document=ReturnDocument.BEFORE, **kwargs):
        common.validate_ok_for_update(update)
        kwargs['update'] = update
        return self.__find_and_modify(filter, projection,
                                      sort, upsert, return_document, **kwargs)
    def save(self, to_save, manipulate=True, check_keys=True, **kwargs):
        warnings.warn("save is deprecated. Use insert_one or replace_one "
                      "instead", DeprecationWarning, stacklevel=2)
        common.validate_is_document_type("to_save", to_save)
        write_concern = None
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        if kwargs:
            write_concern = WriteConcern(**kwargs)
        with self._socket_for_writes() as sock_info:
            if not (isinstance(to_save, RawBSONDocument) or "_id" in to_save):
                return self._insert(sock_info, to_save, True,
                                    check_keys, manipulate, write_concern)
            else:
                self._update(sock_info, {"_id": to_save["_id"]}, to_save, True,
                             check_keys, False, manipulate, write_concern,
                             collation=collation)
                return to_save.get("_id")
    def insert(self, doc_or_docs, manipulate=True,
               check_keys=True, continue_on_error=False, **kwargs):
        warnings.warn("insert is deprecated. Use insert_one or insert_many "
                      "instead.", DeprecationWarning, stacklevel=2)
        write_concern = None
        if kwargs:
            write_concern = WriteConcern(**kwargs)
        with self._socket_for_writes() as sock_info:
            return self._insert(sock_info, doc_or_docs, not continue_on_error,
                                check_keys, manipulate, write_concern)
    def update(self, spec, document, upsert=False, manipulate=False,
               multi=False, check_keys=True, **kwargs):
        warnings.warn("update is deprecated. Use replace_one, update_one or "
                      "update_many instead.", DeprecationWarning, stacklevel=2)
        common.validate_is_mapping("spec", spec)
        common.validate_is_mapping("document", document)
        if document:
            first = next(iter(document))
            if first.startswith('$'):
                check_keys = False
        write_concern = None
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        if kwargs:
            write_concern = WriteConcern(**kwargs)
        with self._socket_for_writes() as sock_info:
            return self._update(sock_info, spec, document, upsert,
                                check_keys, multi, manipulate, write_concern,
                                collation=collation)
    def remove(self, spec_or_id=None, multi=True, **kwargs):
        warnings.warn("remove is deprecated. Use delete_one or delete_many "
                      "instead.", DeprecationWarning, stacklevel=2)
        if spec_or_id is None:
            spec_or_id = {}
        if not isinstance(spec_or_id, collections.Mapping):
            spec_or_id = {"_id": spec_or_id}
        write_concern = None
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        if kwargs:
            write_concern = WriteConcern(**kwargs)
        with self._socket_for_writes() as sock_info:
            return self._delete(sock_info, spec_or_id, multi, write_concern,
                                collation=collation)
    def find_and_modify(self, query={}, update=None,
                        upsert=False, sort=None, full_response=False,
                        manipulate=False, **kwargs):
        warnings.warn("find_and_modify is deprecated, use find_one_and_delete"
                      ", find_one_and_replace, or find_one_and_update instead",
                      DeprecationWarning, stacklevel=2)
        if not update and not kwargs.get('remove', None):
            raise ValueError("Must either update or remove")
        if update and kwargs.get('remove', None):
            raise ValueError("Can't do both update and remove")
        if query:
            kwargs['query'] = query
        if update:
            kwargs['update'] = update
        if upsert:
            kwargs['upsert'] = upsert
        if sort:
            if isinstance(sort, list):
                kwargs['sort'] = helpers._index_document(sort)
            elif (isinstance(sort, _ORDERED_TYPES) or
                  isinstance(sort, dict) and len(sort) == 1):
                warnings.warn("Passing mapping types for `sort` is deprecated,"
                              " use a list of (key, direction) pairs instead",
                              DeprecationWarning, stacklevel=2)
                kwargs['sort'] = sort
            else:
                raise TypeError("sort must be a list of (key, direction) "
                                "pairs, a dict of len 1, or an instance of "
                                "SON or OrderedDict")
        fields = kwargs.pop("fields", None)
        if fields is not None:
            kwargs["fields"] = helpers._fields_list_to_dict(fields, "fields")
        collation = validate_collation_or_none(kwargs.pop('collation', None))
        cmd = SON([("findAndModify", self.__name)])
        cmd.update(kwargs)
        with self._socket_for_writes() as sock_info:
            if sock_info.max_wire_version >= 4 and 'writeConcern' not in cmd:
                wc_doc = self.write_concern.document
                if wc_doc:
                    cmd['writeConcern'] = wc_doc
            out = self._command(sock_info, cmd,
                                read_preference=ReadPreference.PRIMARY,
                                allowable_errors=[_NO_OBJ_ERROR],
                                collation=collation)
            _check_write_command_response([(0, out)])
        if not out['ok']:
            if out["errmsg"] == _NO_OBJ_ERROR:
                return None
            else:
                raise ValueError("Unexpected Error: %s" % (out,))
        if full_response:
            return out
        else:
            document = out.get('value')
            if manipulate:
                document = self.__database._fix_outgoing(document, self)
            return document
    def __iter__(self):
        return self
    def __next__(self):
        raise TypeError("'Collection' object is not iterable")
    next = __next__
    def __call__(self, *args, **kwargs):
        if "." not in self.__name:
            raise TypeError("'Collection' object is not callable. If you "
                            "meant to call the '%s' method on a 'Database' "
                            "object it is failing because no such method "
                            "exists." %
                            self.__name)
        raise TypeError("'Collection' object is not callable. If you meant to "
                        "call the '%s' method on a 'Collection' object it is "
                        "failing because no such method exists." %
                        self.__name.split(".")[-1])
