
class ModelAndInput:
  def get_model(self):
    raise NotImplementedError("must be implemented in descendants")
  def get_data(self):
    raise NotImplementedError("must be implemented in descendants")
  def get_batch_size(self):
    raise NotImplementedError("must be implemented in descendants")
