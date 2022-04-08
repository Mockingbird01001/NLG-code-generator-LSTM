
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from mediapipe.util.sequence import media_sequence_util
msu = media_sequence_util
_HAS_DYNAMIC_ATTRIBUTES = True
EXAMPLE_ID_KEY = "example/id"
EXAMPLE_DATASET_NAME_KEY = "example/dataset_name"
CLIP_DATA_PATH_KEY = "clip/data_path"
CLIP_MEDIA_ID_KEY = "clip/media_id"
ALTERNATIVE_CLIP_MEDIA_ID_KEY = "clip/alternative_media_id"
CLIP_ENCODED_MEDIA_BYTES_KEY = "clip/encoded_media_bytes"
CLIP_ENCODED_MEDIA_START_TIMESTAMP_KEY = "clip/encoded_media_start_timestamp"
CLIP_START_TIMESTAMP_KEY = "clip/start/timestamp"
CLIP_END_TIMESTAMP_KEY = "clip/end/timestamp"
CLIP_LABEL_INDEX_KEY = "clip/label/index"
CLIP_LABEL_STRING_KEY = "clip/label/string"
CLIP_LABEL_CONFIDENCE_KEY = "clip/label/confidence"
msu.create_bytes_context_feature(
    "example_id", EXAMPLE_ID_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "example_dataset_name", EXAMPLE_DATASET_NAME_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "clip_media_id", CLIP_MEDIA_ID_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "clip_alternative_media_id", ALTERNATIVE_CLIP_MEDIA_ID_KEY,
    module_dict=globals())
msu.create_bytes_context_feature(
    "clip_encoded_media_bytes", CLIP_ENCODED_MEDIA_BYTES_KEY,
    module_dict=globals())
msu.create_bytes_context_feature(
    "clip_data_path", CLIP_DATA_PATH_KEY, module_dict=globals())
msu.create_int_context_feature(
    "clip_encoded_media_start_timestamp",
    CLIP_ENCODED_MEDIA_START_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_context_feature(
    "clip_start_timestamp", CLIP_START_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_context_feature(
    "clip_end_timestamp", CLIP_END_TIMESTAMP_KEY, module_dict=globals())
msu.create_bytes_list_context_feature(
    "clip_label_string", CLIP_LABEL_STRING_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "clip_label_index", CLIP_LABEL_INDEX_KEY, module_dict=globals())
msu.create_float_list_context_feature(
    "clip_label_confidence", CLIP_LABEL_CONFIDENCE_KEY, module_dict=globals())
SEGMENT_START_TIMESTAMP_KEY = "segment/start/timestamp"
SEGMENT_START_INDEX_KEY = "segment/start/index"
SEGMENT_END_TIMESTAMP_KEY = "segment/end/timestamp"
SEGMENT_END_INDEX_KEY = "segment/end/index"
SEGMENT_LABEL_INDEX_KEY = "segment/label/index"
SEGMENT_LABEL_STRING_KEY = "segment/label/string"
SEGMENT_LABEL_CONFIDENCE_KEY = "segment/label/confidence"
msu.create_bytes_list_context_feature(
    "segment_label_string", SEGMENT_LABEL_STRING_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_start_timestamp",
    SEGMENT_START_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_start_index", SEGMENT_START_INDEX_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_end_timestamp", SEGMENT_END_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_end_index", SEGMENT_END_INDEX_KEY, module_dict=globals())
msu.create_int_list_context_feature(
    "segment_label_index", SEGMENT_LABEL_INDEX_KEY, module_dict=globals())
msu.create_float_list_context_feature(
    "segment_label_confidence",
    SEGMENT_LABEL_CONFIDENCE_KEY, module_dict=globals())
REGION_BBOX_YMIN_KEY = "region/bbox/ymin"
REGION_BBOX_XMIN_KEY = "region/bbox/xmin"
REGION_BBOX_YMAX_KEY = "region/bbox/ymax"
REGION_BBOX_XMAX_KEY = "region/bbox/xmax"
REGION_POINT_X_KEY = "region/point/x"
REGION_POINT_Y_KEY = "region/point/y"
REGION_RADIUS_KEY = "region/radius"
REGION_3D_POINT_X_KEY = "region/3d_point/x"
REGION_3D_POINT_Y_KEY = "region/3d_point/y"
REGION_3D_POINT_Z_KEY = "region/3d_point/z"
REGION_NUM_REGIONS_KEY = "region/num_regions"
REGION_IS_ANNOTATED_KEY = "region/is_annotated"
REGION_IS_GENERATED_KEY = "region/is_generated"
REGION_IS_OCCLUDED_KEY = "region/is_occluded"
REGION_LABEL_INDEX_KEY = "region/label/index"
REGION_LABEL_STRING_KEY = "region/label/string"
REGION_LABEL_CONFIDENCE_KEY = "region/label/confidence"
REGION_TRACK_INDEX_KEY = "region/track/index"
REGION_TRACK_STRING_KEY = "region/track/string"
REGION_TRACK_CONFIDENCE_KEY = "region/track/confidence"
REGION_CLASS_INDEX_KEY = "region/class/index"
REGION_CLASS_STRING_KEY = "region/class/string"
REGION_CLASS_CONFIDENCE_KEY = "region/class/confidence"
REGION_TIMESTAMP_KEY = "region/timestamp"
REGION_UNMODIFIED_TIMESTAMP_KEY = "region/unmodified_timestamp"
REGION_PARTS_KEY = "region/parts"
REGION_EMBEDDING_DIMENSIONS_PER_REGION_KEY = (
    "region/embedding/dimensions_per_region")
REGION_EMBEDDING_FORMAT_KEY = "region/embedding/format"
REGION_EMBEDDING_FLOAT_KEY = "region/embedding/float"
REGION_EMBEDDING_ENCODED_KEY = "region/embedding/encoded"
REGION_EMBEDDING_CONFIDENCE_KEY = "region/embedding/confidence"
def _create_region_with_prefix(name, prefix):
  msu.create_int_feature_list(name + "_num_regions", REGION_NUM_REGIONS_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(name + "_is_annotated", REGION_IS_ANNOTATED_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_is_occluded", REGION_IS_OCCLUDED_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_is_generated", REGION_IS_GENERATED_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(name + "_timestamp", REGION_TIMESTAMP_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(
      name + "_unmodified_timestamp", REGION_UNMODIFIED_TIMESTAMP_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(
      name + "_label_string", REGION_LABEL_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_label_index", REGION_LABEL_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_label_confidence", REGION_LABEL_CONFIDENCE_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(
      name + "_class_string", REGION_CLASS_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_class_index", REGION_CLASS_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_class_confidence", REGION_CLASS_CONFIDENCE_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(
      name + "_track_string", REGION_TRACK_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_feature_list(
      name + "_track_index", REGION_TRACK_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_track_confidence", REGION_TRACK_CONFIDENCE_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_ymin", REGION_BBOX_YMIN_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_xmin", REGION_BBOX_XMIN_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_ymax", REGION_BBOX_YMAX_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_xmax", REGION_BBOX_XMAX_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_point_x", REGION_POINT_X_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_point_y", REGION_POINT_Y_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_3d_point_x", REGION_3D_POINT_X_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_3d_point_y", REGION_3D_POINT_Y_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(
      name + "_3d_point_z", REGION_3D_POINT_Z_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_list_context_feature(name + "_parts",
                                        REGION_PARTS_KEY,
                                        prefix=prefix, module_dict=globals())
  msu.create_float_list_context_feature(
      name + "_embedding_dimensions_per_region",
      REGION_EMBEDDING_DIMENSIONS_PER_REGION_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_context_feature(name + "_embedding_format",
                                   REGION_EMBEDDING_FORMAT_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_embedding_floats",
                                     REGION_EMBEDDING_FLOAT_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(name + "_embedding_encoded",
                                     REGION_EMBEDDING_ENCODED_KEY,
                                     prefix=prefix, module_dict=globals())
  msu.create_float_list_feature_list(name + "_embedding_confidence",
                                     REGION_EMBEDDING_CONFIDENCE_KEY,
                                     prefix=prefix, module_dict=globals())
  def get_prefixed_bbox_at(index, sequence_example, prefix):
    return np.stack((
        get_bbox_ymin_at(index, sequence_example, prefix=prefix),
        get_bbox_xmin_at(index, sequence_example, prefix=prefix),
        get_bbox_ymax_at(index, sequence_example, prefix=prefix),
        get_bbox_xmax_at(index, sequence_example, prefix=prefix)),
                    1)
  def add_prefixed_bbox(values, sequence_example, prefix):
    values = np.array(values)
    if values.size == 0:
      add_bbox_ymin([], sequence_example, prefix=prefix)
      add_bbox_xmin([], sequence_example, prefix=prefix)
      add_bbox_ymax([], sequence_example, prefix=prefix)
      add_bbox_xmax([], sequence_example, prefix=prefix)
    else:
      add_bbox_ymin(values[:, 0], sequence_example, prefix=prefix)
      add_bbox_xmin(values[:, 1], sequence_example, prefix=prefix)
      add_bbox_ymax(values[:, 2], sequence_example, prefix=prefix)
      add_bbox_xmax(values[:, 3], sequence_example, prefix=prefix)
  def get_prefixed_bbox_size(sequence_example, prefix):
    return get_bbox_ymin_size(sequence_example, prefix=prefix)
  def has_prefixed_bbox(sequence_example, prefix):
    return has_bbox_ymin(sequence_example, prefix=prefix)
  def clear_prefixed_bbox(sequence_example, prefix):
    clear_bbox_ymin(sequence_example, prefix=prefix)
    clear_bbox_xmin(sequence_example, prefix=prefix)
    clear_bbox_ymax(sequence_example, prefix=prefix)
    clear_bbox_xmax(sequence_example, prefix=prefix)
  def get_prefixed_point_at(index, sequence_example, prefix):
    return np.stack((
        get_bbox_point_y_at(index, sequence_example, prefix=prefix),
        get_bbox_point_x_at(index, sequence_example, prefix=prefix)),
                    1)
  def add_prefixed_point(values, sequence_example, prefix):
    add_bbox_point_y(values[:, 0], sequence_example, prefix=prefix)
    add_bbox_point_x(values[:, 1], sequence_example, prefix=prefix)
  def get_prefixed_point_size(sequence_example, prefix):
    return get_bbox_point_y_size(sequence_example, prefix=prefix)
  def has_prefixed_point(sequence_example, prefix):
    return has_bbox_point_y(sequence_example, prefix=prefix)
  def clear_prefixed_point(sequence_example, prefix):
    clear_bbox_point_y(sequence_example, prefix=prefix)
    clear_bbox_point_x(sequence_example, prefix=prefix)
  def get_prefixed_3d_point_at(index, sequence_example, prefix):
    return np.stack((
        get_bbox_3d_point_x_at(index, sequence_example, prefix=prefix),
        get_bbox_3d_point_y_at(index, sequence_example, prefix=prefix),
        get_bbox_3d_point_z_at(index, sequence_example, prefix=prefix)),
                    1)
  def add_prefixed_3d_point(values, sequence_example, prefix):
    add_bbox_3d_point_x(values[:, 0], sequence_example, prefix=prefix)
    add_bbox_3d_point_y(values[:, 1], sequence_example, prefix=prefix)
    add_bbox_3d_point_z(values[:, 2], sequence_example, prefix=prefix)
  def get_prefixed_3d_point_size(sequence_example, prefix):
    return get_bbox_3d_point_x_size(sequence_example, prefix=prefix)
  def has_prefixed_3d_point(sequence_example, prefix):
    return has_bbox_3d_point_x(sequence_example, prefix=prefix)
  def clear_prefixed_3d_point(sequence_example, prefix):
    clear_bbox_3d_point_x(sequence_example, prefix=prefix)
    clear_bbox_3d_point_y(sequence_example, prefix=prefix)
    clear_bbox_3d_point_z(sequence_example, prefix=prefix)
  msu.add_functions_to_module({
      "get_" + name + "_at":
          msu.function_with_default(get_prefixed_bbox_at, prefix),
      "add_" + name:
          msu.function_with_default(add_prefixed_bbox, prefix),
      "get_" + name + "_size":
          msu.function_with_default(get_prefixed_bbox_size, prefix),
      "has_" + name:
          msu.function_with_default(has_prefixed_bbox, prefix),
      "clear_" + name:
          msu.function_with_default(clear_prefixed_bbox, prefix),
  }, module_dict=globals())
  msu.add_functions_to_module({
      "get_" + name + "_point_at":
          msu.function_with_default(get_prefixed_point_at, prefix),
      "add_" + name + "_point":
          msu.function_with_default(add_prefixed_point, prefix),
      "get_" + name + "_point_size":
          msu.function_with_default(get_prefixed_point_size, prefix),
      "has_" + name + "_point":
          msu.function_with_default(has_prefixed_point, prefix),
      "clear_" + name + "_point":
          msu.function_with_default(clear_prefixed_point, prefix),
  }, module_dict=globals())
  msu.add_functions_to_module({
      "get_" + name + "_3d_point_at":
          msu.function_with_default(get_prefixed_3d_point_at, prefix),
      "add_" + name + "_3d_point":
          msu.function_with_default(add_prefixed_3d_point, prefix),
      "get_" + name + "_3d_point_size":
          msu.function_with_default(get_prefixed_3d_point_size, prefix),
      "has_" + name + "_3d_point":
          msu.function_with_default(has_prefixed_3d_point, prefix),
      "clear_" + name + "_3d_point":
          msu.function_with_default(clear_prefixed_3d_point, prefix),
  }, module_dict=globals())
PREDICTED_PREFIX = "PREDICTED"
_create_region_with_prefix("bbox", "")
_create_region_with_prefix("predicted_bbox", PREDICTED_PREFIX)
IMAGE_FORMAT_KEY = "image/format"
IMAGE_CHANNELS_KEY = "image/channels"
IMAGE_COLORSPACE_KEY = "image/colorspace"
IMAGE_HEIGHT_KEY = "image/height"
IMAGE_WIDTH_KEY = "image/width"
IMAGE_FRAME_RATE_KEY = "image/frame_rate"
IMAGE_SATURATION_KEY = "image/saturation"
IMAGE_CLASS_LABEL_INDEX_KEY = "image/class/label/index"
IMAGE_CLASS_LABEL_STRING_KEY = "image/class/label/string"
IMAGE_OBJECT_CLASS_INDEX_KEY = "image/object/class/index"
IMAGE_ENCODED_KEY = "image/encoded"
IMAGE_MULTI_ENCODED_KEY = "image/multi_encoded"
IMAGE_TIMESTAMP_KEY = "image/timestamp"
IMAGE_LABEL_INDEX_KEY = "image/label/index"
IMAGE_LABEL_STRING_KEY = "image/label/string"
IMAGE_LABEL_CONFIDENCE_KEY = "image/label/confidence"
IMAGE_DATA_PATH_KEY = "image/data_path"
def _create_image_with_prefix(name, prefix):
  msu.create_bytes_context_feature(name + "_format", IMAGE_FORMAT_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_bytes_context_feature(name + "_colorspace", IMAGE_COLORSPACE_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_int_context_feature(name + "_channels", IMAGE_CHANNELS_KEY,
                                 prefix=prefix, module_dict=globals())
  msu.create_int_context_feature(name + "_height", IMAGE_HEIGHT_KEY,
                                 prefix=prefix, module_dict=globals())
  msu.create_int_context_feature(name + "_width", IMAGE_WIDTH_KEY,
                                 prefix=prefix, module_dict=globals())
  msu.create_bytes_feature_list(name + "_encoded", IMAGE_ENCODED_KEY,
                                prefix=prefix, module_dict=globals())
  msu.create_float_context_feature(name + "_frame_rate", IMAGE_FRAME_RATE_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_bytes_list_context_feature(
      name + "_class_label_string", IMAGE_CLASS_LABEL_STRING_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_context_feature(
      name + "_class_label_index", IMAGE_CLASS_LABEL_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_int_list_context_feature(
      name + "_object_class_index", IMAGE_OBJECT_CLASS_INDEX_KEY,
      prefix=prefix, module_dict=globals())
  msu.create_bytes_context_feature(name + "_data_path", IMAGE_DATA_PATH_KEY,
                                   prefix=prefix, module_dict=globals())
  msu.create_int_feature_list(name + "_timestamp", IMAGE_TIMESTAMP_KEY,
                              prefix=prefix, module_dict=globals())
  msu.create_bytes_list_feature_list(name + "_multi_encoded",
                                     IMAGE_MULTI_ENCODED_KEY, prefix=prefix,
                                     module_dict=globals())
FORWARD_FLOW_PREFIX = "FORWARD_FLOW"
CLASS_SEGMENTATION_PREFIX = "CLASS_SEGMENTATION"
INSTANCE_SEGMENTATION_PREFIX = "INSTANCE_SEGMENTATION"
_create_image_with_prefix("image", "")
_create_image_with_prefix("forward_flow", FORWARD_FLOW_PREFIX)
_create_image_with_prefix("class_segmentation", CLASS_SEGMENTATION_PREFIX)
_create_image_with_prefix("instance_segmentation", INSTANCE_SEGMENTATION_PREFIX)
TEXT_LANGUAGE_KEY = "text/language"
TEXT_CONTEXT_CONTENT_KEY = "text/context/content"
TEXT_CONTENT_KEY = "text/content"
TEXT_TIMESTAMP_KEY = "text/timestamp"
TEXT_DURATION_KEY = "text/duration"
TEXT_CONFIDENCE_KEY = "text/confidence"
TEXT_EMBEDDING_KEY = "text/embedding"
TEXT_TOKEN_ID_KEY = "text/token/id"
msu.create_bytes_context_feature(
    "text_language", TEXT_LANGUAGE_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "text_context_content", TEXT_CONTEXT_CONTENT_KEY, module_dict=globals())
msu.create_bytes_feature_list(
    "text_content", TEXT_CONTENT_KEY, module_dict=globals())
msu.create_int_feature_list(
    "text_timestamp", TEXT_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_feature_list(
    "text_duration", TEXT_DURATION_KEY, module_dict=globals())
msu.create_float_feature_list(
    "text_confidence", TEXT_CONFIDENCE_KEY, module_dict=globals())
msu.create_float_list_feature_list(
    "text_embedding", TEXT_EMBEDDING_KEY, module_dict=globals())
msu.create_int_feature_list(
    "text_token_id", TEXT_TOKEN_ID_KEY, module_dict=globals())
FEATURE_DIMENSIONS_KEY = "feature/dimensions"
FEATURE_RATE_KEY = "feature/rate"
FEATURE_BYTES_FORMAT_KEY = "feature/bytes/format"
FEATURE_SAMPLE_RATE_KEY = "feature/sample_rate"
FEATURE_NUM_CHANNELS_KEY = "feature/num_channels"
FEATURE_NUM_SAMPLES_KEY = "feature/num_samples"
FEATURE_PACKET_RATE_KEY = "feature/packet_rate"
FEATURE_AUDIO_SAMPLE_RATE_KEY = "feature/audio_sample_rate"
FEATURE_FLOATS_KEY = "feature/floats"
FEATURE_BYTES_KEY = "feature/bytes"
FEATURE_INTS_KEY = "feature/ints"
FEATURE_TIMESTAMP_KEY = "feature/timestamp"
FEATURE_DURATION_KEY = "feature/duration"
FEATURE_CONFIDENCE_KEY = "feature/confidence"
msu.create_int_list_context_feature(
    "feature_dimensions", FEATURE_DIMENSIONS_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_rate", FEATURE_RATE_KEY, module_dict=globals())
msu.create_bytes_context_feature(
    "feature_bytes_format", FEATURE_BYTES_FORMAT_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_sample_rate", FEATURE_SAMPLE_RATE_KEY, module_dict=globals())
msu.create_int_context_feature(
    "feature_num_channels", FEATURE_NUM_CHANNELS_KEY, module_dict=globals())
msu.create_int_context_feature(
    "feature_num_samples", FEATURE_NUM_SAMPLES_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_packet_rate", FEATURE_PACKET_RATE_KEY, module_dict=globals())
msu.create_float_context_feature(
    "feature_audio_sample_rate", FEATURE_AUDIO_SAMPLE_RATE_KEY,
    module_dict=globals())
msu.create_float_list_feature_list(
    "feature_floats", FEATURE_FLOATS_KEY, module_dict=globals())
msu.create_bytes_list_feature_list(
    "feature_bytes", FEATURE_BYTES_KEY, module_dict=globals())
msu.create_int_list_feature_list(
    "feature_ints", FEATURE_INTS_KEY, module_dict=globals())
msu.create_int_feature_list(
    "feature_timestamp", FEATURE_TIMESTAMP_KEY, module_dict=globals())
msu.create_int_list_feature_list(
    "feature_duration", FEATURE_DURATION_KEY, module_dict=globals())
msu.create_float_list_feature_list(
    "feature_confidence", FEATURE_CONFIDENCE_KEY, module_dict=globals())
