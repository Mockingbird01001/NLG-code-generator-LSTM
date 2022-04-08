
def dismantle_ordered_dict(ordered_dict):
  if problematic_cycle:
    try:
      del problematic_cycle[0][:]
    except TypeError:
      pass
