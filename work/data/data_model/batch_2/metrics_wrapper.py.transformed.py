
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.metrics import converter_error_data_pb2
def retrieve_collected_errors():
  serialized_message_list = wrap_toco.wrapped_retrieve_collected_errors()
  return list(
      map(converter_error_data_pb2.ConverterErrorData.FromString,
          serialized_message_list))
