
from tensorflow.python.ops import gen_collective_ops
def all_reduce(t,
               group_size,
               group_key,
               instance_key,
               merge_op='Add',
               final_op='Id',
               subdiv_offsets=(0,),
               communication_hint='auto',
               timeout=0):
  if group_size < 1:
    raise ValueError('Parameter `group_size` to all_reduce must be at least 1. '
                     f'Received: {group_size}.')
  return gen_collective_ops.collective_reduce(
      t,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      merge_op=merge_op,
      final_op=final_op,
      subdiv_offsets=subdiv_offsets,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)
def assign_group_v2(group_assignment, device_index, base_key):
  group_size, group_key = gen_collective_ops.collective_assign_group_v2(
      group_assignment=group_assignment,
      device_index=device_index,
      base_key=base_key)
  return group_size, group_key
def all_reduce_v2(t,
                  group_size,
                  group_key,
                  instance_key,
                  merge_op='Add',
                  final_op='Id',
                  communication_hint='auto',
                  timeout=0,
                  ordering_token=None,
                  max_subdivs_per_device=-1,
                  name=None):
  if ordering_token is not None:
    ordering_token = [ordering_token]
  else:
    ordering_token = []
  return gen_collective_ops.collective_reduce_v2(
      t,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      merge_op=merge_op,
      final_op=final_op,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout,
      ordering_token=ordering_token,
      max_subdivs_per_device=max_subdivs_per_device,
      name=name)
def all_gather(t,
               group_size,
               group_key,
               instance_key,
               communication_hint='auto',
               timeout=0):
  if group_size < 1:
    raise ValueError('Parameter `group_size` to all_gather must be at least 1.'
                     f' Received: {group_size}.')
  return gen_collective_ops.collective_gather(
      t,
      shape=[0],
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)
def all_gather_v2(t,
                  group_size,
                  group_key,
                  instance_key,
                  communication_hint='auto',
                  timeout=0,
                  ordering_token=None,
                  name=None):
  if ordering_token is not None:
    ordering_token = [ordering_token]
  else:
    ordering_token = []
  return gen_collective_ops.collective_gather_v2(
      t,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout,
      ordering_token=ordering_token,
      name=name)
def broadcast_send(t,
                   shape,
                   dtype,
                   group_size,
                   group_key,
                   instance_key,
                   communication_hint='auto',
                   timeout=0):
  if group_size <= 1:
    raise ValueError(
        'Parameter `group_size` to broadcast_send must be at least 2. '
        f'Received: {group_size}.')
  if t.shape != shape:
    raise ValueError(
        'Shape of broadcast_send tensor `t` not equal to declared shape. '
        f'Received {t.shape}, expected {shape}.')
  if t.dtype != dtype:
    raise ValueError(
        'Type of broadcast_send tensor `t` not equal to declared type. '
        f'Received {t.dtype}, expected {dtype}.')
  return gen_collective_ops.collective_bcast_send(
      t,
      shape=shape,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)
def broadcast_send_v2(t,
                      group_size,
                      group_key,
                      instance_key,
                      communication_hint='auto',
                      timeout=0):
  return gen_collective_ops.collective_bcast_send_v2(
      t,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)
def broadcast_recv(shape,
                   dtype,
                   group_size,
                   group_key,
                   instance_key,
                   communication_hint='auto',
                   timeout=0):
  if group_size <= 1:
    raise ValueError(
        'Parameter `group_size` to broadcast_send must be at least 2. '
        f'Received: {group_size}.')
  return gen_collective_ops.collective_bcast_recv(
      shape=shape,
      T=dtype,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)
def broadcast_recv_v2(shape,
                      dtype,
                      group_size,
                      group_key,
                      instance_key,
                      communication_hint='auto',
                      timeout=0):
  return gen_collective_ops.collective_bcast_recv_v2(
      T=dtype,
      group_size=group_size,
      group_key=group_key,
      instance_key=instance_key,
      shape=shape,
      communication_hint=communication_hint.lower(),
      timeout_seconds=timeout)
def initialize_communicator(group_key,
                            rank,
                            group_size,
                            communication_hint='auto',
                            timeout_seconds=0):
  return gen_collective_ops.collective_initialize_communicator(
      group_key=group_key,
      rank=rank,
      group_size=group_size,
      communication_hint=communication_hint,
      timeout_seconds=timeout_seconds)
def all_reduce_v3(communicator,
                  t,
                  reduction='Add',
                  group_assignment=None,
                  timeout_seconds=None):
  if group_assignment is None:
    group_assignment = []
  return gen_collective_ops.collective_reduce_v3(
      communicator=communicator,
      input=t,
      group_assignment=group_assignment,
      reduction=reduction,
      timeout_seconds=timeout_seconds)
def all_to_all_v3(communicator, t, group_assignment=None, timeout_seconds=None):
  if group_assignment is None:
    group_assignment = []
  return gen_collective_ops.collective_all_to_all_v3(
      communicator=communicator,
      input=t,
      group_assignment=group_assignment,
      timeout_seconds=timeout_seconds)
