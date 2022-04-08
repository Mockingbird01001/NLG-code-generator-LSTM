
import grpc
from tensorflow.core.profiler import profiler_analysis_pb2 as third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
class ProfileAnalysisStub(object):
  def __init__(self, channel):
    self.NewSession = channel.unary_unary(
        '/tensorflow.ProfileAnalysis/NewSession',
        request_serializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
        .NewProfileSessionRequest.SerializeToString,
        response_deserializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
        .NewProfileSessionResponse.FromString,
    )
    self.EnumSessions = channel.unary_unary(
        '/tensorflow.ProfileAnalysis/EnumSessions',
        request_serializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
        .EnumProfileSessionsAndToolsRequest.SerializeToString,
        response_deserializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
        .EnumProfileSessionsAndToolsResponse.FromString,
    )
    self.GetSessionToolData = channel.unary_unary(
        '/tensorflow.ProfileAnalysis/GetSessionToolData',
        request_serializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
        .ProfileSessionDataRequest.SerializeToString,
        response_deserializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
        .ProfileSessionDataResponse.FromString,
    )
class ProfileAnalysisServicer(object):
  def NewSession(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')
  def EnumSessions(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')
  def GetSessionToolData(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')
def add_ProfileAnalysisServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'NewSession':
          grpc.unary_unary_rpc_method_handler(
              servicer.NewSession,
              request_deserializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
              .NewProfileSessionRequest.FromString,
              response_serializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
              .NewProfileSessionResponse.SerializeToString,
          ),
      'EnumSessions':
          grpc.unary_unary_rpc_method_handler(
              servicer.EnumSessions,
              request_deserializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
              .EnumProfileSessionsAndToolsRequest.FromString,
              response_serializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
              .EnumProfileSessionsAndToolsResponse.SerializeToString,
          ),
      'GetSessionToolData':
          grpc.unary_unary_rpc_method_handler(
              servicer.GetSessionToolData,
              request_deserializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
              .ProfileSessionDataRequest.FromString,
              response_serializer=third__party_dot_tensorflow_dot_core_dot_profiler_dot_profiler__analysis__pb2
              .ProfileSessionDataResponse.SerializeToString,
          ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.ProfileAnalysis', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
