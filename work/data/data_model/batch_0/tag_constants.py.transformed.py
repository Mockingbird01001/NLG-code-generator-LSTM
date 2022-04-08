
from tensorflow.python.util.tf_export import tf_export
SERVING = "serve"
tf_export(
    "saved_model.SERVING",
    v1=["saved_model.SERVING",
        "saved_model.tag_constants.SERVING"]).export_constant(
            __name__, "SERVING")
TRAINING = "train"
tf_export(
    "saved_model.TRAINING",
    v1=["saved_model.TRAINING",
        "saved_model.tag_constants.TRAINING"]).export_constant(
            __name__, "TRAINING")
EVAL = "eval"
GPU = "gpu"
tf_export(
    "saved_model.GPU", v1=["saved_model.GPU",
                           "saved_model.tag_constants.GPU"]).export_constant(
                               __name__, "GPU")
TPU = "tpu"
tf_export(
    "saved_model.TPU", v1=["saved_model.TPU",
                           "saved_model.tag_constants.TPU"]).export_constant(
                               __name__, "TPU")
