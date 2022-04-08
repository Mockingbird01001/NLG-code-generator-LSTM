
from tensorflow.python.autograph.core import config_lib
Action = config_lib.Action
Convert = config_lib.Convert
DoNotConvert = config_lib.DoNotConvert
CONVERSION_RULES = (
    Convert('tensorflow.python.training.experimental'),
    DoNotConvert('collections'),
    DoNotConvert('copy'),
    DoNotConvert('cProfile'),
    DoNotConvert('inspect'),
    DoNotConvert('ipdb'),
    DoNotConvert('linecache'),
    DoNotConvert('mock'),
    DoNotConvert('pathlib'),
    DoNotConvert('pdb'),
    DoNotConvert('posixpath'),
    DoNotConvert('pstats'),
    DoNotConvert('re'),
    DoNotConvert('threading'),
    DoNotConvert('urllib'),
    DoNotConvert('matplotlib'),
    DoNotConvert('numpy'),
    DoNotConvert('pandas'),
    DoNotConvert('tensorflow'),
    DoNotConvert('PIL'),
    DoNotConvert('absl.logging'),
    DoNotConvert('tensorflow_probability'),
    DoNotConvert('tensorflow_datasets.core'),
    DoNotConvert('keras'),
)
