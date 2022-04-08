
import collections
_DeferredSlotVariableRestoration = collections.namedtuple(
    "_DeferredSlotVariableRestoration", [
        "original_variable",
        "slot_variable_id",
        "slot_name",
    ])
def queue_slot_variables(checkpoint_position, visit_queue):
  trackable = checkpoint_position.trackable
  checkpoint = checkpoint_position.checkpoint
  for deferred_slot_restoration in (
      checkpoint.deferred_slot_restorations.pop(checkpoint_position.proto_id,
                                                ())):
    slot_variable_position, slot_variable = (
        checkpoint_position.create_slot_variable_position(
            trackable, deferred_slot_restoration.original_variable,
            deferred_slot_restoration.slot_variable_id,
            deferred_slot_restoration.slot_name))
    if slot_variable_position is not None:
      visit_queue.append((slot_variable_position, slot_variable))
  for slot_restoration in checkpoint.slot_restorations.pop(
      checkpoint_position.proto_id, ()):
    optimizer_object = checkpoint.object_by_proto_id.get(
        slot_restoration.optimizer_id, None)
    if optimizer_object is None:
      checkpoint.deferred_slot_restorations.setdefault(
          slot_restoration.optimizer_id, []).append(
              _DeferredSlotVariableRestoration(
                  original_variable=trackable,
                  slot_variable_id=slot_restoration.slot_variable_id,
                  slot_name=slot_restoration.slot_name))
    elif hasattr(optimizer_object, "_create_or_restore_slot_variable"):
      slot_variable_position, slot_variable = (
          checkpoint_position.create_slot_variable_position(
              optimizer_object, trackable, slot_restoration.slot_variable_id,
              slot_restoration.slot_name))
      if slot_variable_position is not None:
        visit_queue.append((slot_variable_position, slot_variable))
