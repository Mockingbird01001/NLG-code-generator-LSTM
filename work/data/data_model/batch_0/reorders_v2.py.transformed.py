
reorders = {
    'tf.argmax': ['input', 'axis', 'name', 'dimension', 'output_type'],
    'tf.argmin': ['input', 'axis', 'name', 'dimension', 'output_type'],
    'tf.batch_to_space': ['input', 'crops', 'block_size', 'name', 'block_shape'],
    'tf.boolean_mask': ['tensor', 'mask', 'name', 'axis'],
    'tf.cond': ['pred', 'true_fn', 'false_fn', 'strict', 'name', 'fn1', 'fn2'],
    'tf.confusion_matrix': ['labels', 'predictions', 'num_classes', 'dtype', 'name', 'weights'],
    'tf.convert_to_tensor': ['value', 'dtype', 'name', 'preferred_dtype', 'dtype_hint'],
    'tf.data.experimental.RaggedTensorStructure': ['dtype', 'shape', 'ragged_rank'],
    'tf.data.experimental.SparseTensorStructure': ['dtype', 'shape'],
    'tf.data.experimental.TensorArrayStructure': ['dtype', 'element_shape', 'dynamic_size', 'infer_shape'],
    'tf.data.experimental.TensorStructure': ['dtype', 'shape'],
    'tf.decode_csv': ['records', 'record_defaults', 'field_delim', 'use_quote_delim', 'name', 'na_value', 'select_cols'],
    'tf.depth_to_space': ['input', 'block_size', 'name', 'data_format'],
    'tf.estimator.BaselineClassifier': ['model_dir', 'n_classes', 'weight_column', 'label_vocabulary', 'optimizer', 'config', 'loss_reduction'],
    'tf.estimator.BaselineRegressor': ['model_dir', 'label_dimension', 'weight_column', 'optimizer', 'config', 'loss_reduction'],
    'tf.estimator.DNNClassifier': ['hidden_units', 'feature_columns', 'model_dir', 'n_classes', 'weight_column', 'label_vocabulary', 'optimizer', 'activation_fn', 'dropout', 'input_layer_partitioner', 'config', 'warm_start_from', 'loss_reduction', 'batch_norm'],
    'tf.estimator.DNNLinearCombinedClassifier': ['model_dir', 'linear_feature_columns', 'linear_optimizer', 'dnn_feature_columns', 'dnn_optimizer', 'dnn_hidden_units', 'dnn_activation_fn', 'dnn_dropout', 'n_classes', 'weight_column', 'label_vocabulary', 'input_layer_partitioner', 'config', 'warm_start_from', 'loss_reduction', 'batch_norm', 'linear_sparse_combiner'],
    'tf.estimator.DNNLinearCombinedRegressor': ['model_dir', 'linear_feature_columns', 'linear_optimizer', 'dnn_feature_columns', 'dnn_optimizer', 'dnn_hidden_units', 'dnn_activation_fn', 'dnn_dropout', 'label_dimension', 'weight_column', 'input_layer_partitioner', 'config', 'warm_start_from', 'loss_reduction', 'batch_norm', 'linear_sparse_combiner'],
    'tf.estimator.DNNRegressor': ['hidden_units', 'feature_columns', 'model_dir', 'label_dimension', 'weight_column', 'optimizer', 'activation_fn', 'dropout', 'input_layer_partitioner', 'config', 'warm_start_from', 'loss_reduction', 'batch_norm'],
    'tf.estimator.LinearClassifier': ['feature_columns', 'model_dir', 'n_classes', 'weight_column', 'label_vocabulary', 'optimizer', 'config', 'partitioner', 'warm_start_from', 'loss_reduction', 'sparse_combiner'],
    'tf.estimator.LinearRegressor': ['feature_columns', 'model_dir', 'label_dimension', 'weight_column', 'optimizer', 'config', 'partitioner', 'warm_start_from', 'loss_reduction', 'sparse_combiner'],
    'tf.feature_column.categorical_column_with_vocabulary_file': ['key', 'vocabulary_file', 'vocabulary_size', 'num_oov_buckets', 'default_value', 'dtype'],
    'tf.gradients': ['ys', 'xs', 'grad_ys', 'name', 'colocate_gradients_with_ops', 'gate_gradients', 'aggregation_method', 'stop_gradients', 'unconnected_gradients'],
    'tf.hessians': ['ys', 'xs', 'name', 'colocate_gradients_with_ops', 'gate_gradients', 'aggregation_method'],
    'tf.image.sample_distorted_bounding_box': ['image_size', 'bounding_boxes', 'seed', 'seed2', 'min_object_covered', 'aspect_ratio_range', 'area_range', 'max_attempts', 'use_image_if_no_bounding_boxes', 'name'],
    'tf.initializers.uniform_unit_scaling': ['factor', 'seed', 'dtype'],
    'tf.io.decode_csv': ['records', 'record_defaults', 'field_delim', 'use_quote_delim', 'name', 'na_value', 'select_cols'],
    'tf.io.parse_example': ['serialized', 'features', 'name', 'example_names'],
    'tf.io.parse_single_example': ['serialized', 'features', 'name', 'example_names'],
    'tf.io.serialize_many_sparse': ['sp_input', 'name', 'out_type'],
    'tf.io.serialize_sparse': ['sp_input', 'name', 'out_type'],
    'tf.linalg.norm': ['tensor', 'ord', 'axis', 'keepdims', 'name', 'keep_dims'],
    'tf.math.argmax': ['input', 'axis', 'name', 'dimension', 'output_type'],
    'tf.math.argmin': ['input', 'axis', 'name', 'dimension', 'output_type'],
    'tf.math.confusion_matrix': ['labels', 'predictions', 'num_classes', 'dtype', 'name', 'weights'],
    'tf.math.in_top_k': ['predictions', 'targets', 'k', 'name'],
    'tf.math.reduce_all': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_any': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_logsumexp': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_max': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_mean': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_min': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_prod': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.math.reduce_sum': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.multinomial': ['logits', 'num_samples', 'seed', 'name', 'output_dtype'],
    'tf.nn.avg_pool': ['value', 'ksize', 'strides', 'padding', 'data_format', 'name', 'input'],
    'tf.nn.avg_pool2d': ['value', 'ksize', 'strides', 'padding', 'data_format', 'name', 'input'],
    'tf.nn.conv1d': ['value', 'filters', 'stride', 'padding', 'use_cudnn_on_gpu', 'data_format', 'name', 'input', 'dilations'],
    'tf.nn.conv2d': ['input', 'filter', 'strides', 'padding', 'use_cudnn_on_gpu', 'data_format', 'dilations', 'name', 'filters'],
    'tf.nn.conv2d_backprop_input': ['input_sizes', 'filter', 'out_backprop', 'strides', 'padding', 'use_cudnn_on_gpu', 'data_format', 'dilations', 'name', 'filters'],
    'tf.nn.convolution': ['input', 'filter', 'padding', 'strides', 'dilation_rate', 'name', 'data_format', 'filters', 'dilations'],
    'tf.nn.crelu': ['features', 'name', 'axis'],
    'tf.nn.ctc_beam_search_decoder': ['inputs', 'sequence_length', 'beam_width', 'top_paths', 'merge_repeated'],
    'tf.nn.depth_to_space': ['input', 'block_size', 'name', 'data_format'],
    'tf.nn.depthwise_conv2d': ['input', 'filter', 'strides', 'padding', 'rate', 'name', 'data_format', 'dilations'],
    'tf.nn.embedding_lookup': ['params', 'ids', 'partition_strategy', 'name', 'validate_indices', 'max_norm'],
    'tf.nn.embedding_lookup_sparse': ['params', 'sp_ids', 'sp_weights', 'partition_strategy', 'name', 'combiner', 'max_norm'],
    'tf.nn.fractional_avg_pool': ['value', 'pooling_ratio', 'pseudo_random', 'overlapping', 'deterministic', 'seed', 'seed2', 'name'],
    'tf.nn.fractional_max_pool': ['value', 'pooling_ratio', 'pseudo_random', 'overlapping', 'deterministic', 'seed', 'seed2', 'name'],
    'tf.nn.in_top_k': ['predictions', 'targets', 'k', 'name'],
    'tf.nn.max_pool': ['value', 'ksize', 'strides', 'padding', 'data_format', 'name', 'input'],
    'tf.nn.moments': ['x', 'axes', 'shift', 'name', 'keep_dims', 'keepdims'],
    'tf.nn.pool': ['input', 'window_shape', 'pooling_type', 'padding', 'dilation_rate', 'strides', 'name', 'data_format', 'dilations'],
    'tf.nn.separable_conv2d': ['input', 'depthwise_filter', 'pointwise_filter', 'strides', 'padding', 'rate', 'name', 'data_format', 'dilations'],
    'tf.nn.softmax_cross_entropy_with_logits': ['_sentinel', 'labels', 'logits', 'dim', 'name', 'axis'],
    'tf.nn.space_to_batch': ['input', 'paddings', 'block_size', 'name', 'block_shape'],
    'tf.nn.space_to_depth': ['input', 'block_size', 'name', 'data_format'],
    'tf.nn.weighted_moments': ['x', 'axes', 'frequency_weights', 'name', 'keep_dims', 'keepdims'],
    'tf.norm': ['tensor', 'ord', 'axis', 'keepdims', 'name', 'keep_dims'],
    'tf.pad': ['tensor', 'paddings', 'mode', 'name', 'constant_values'],
    'tf.parse_example': ['serialized', 'features', 'name', 'example_names'],
    'tf.parse_single_example': ['serialized', 'features', 'name', 'example_names'],
    'tf.quantize_v2': ['input', 'min_range', 'max_range', 'T', 'mode', 'name', 'round_mode', 'narrow_range', 'axis', 'ensure_minimum_range'],
    'tf.random.multinomial': ['logits', 'num_samples', 'seed', 'name', 'output_dtype'],
    'tf.random.poisson': ['lam', 'shape', 'dtype', 'seed', 'name'],
    'tf.random_poisson': ['lam', 'shape', 'dtype', 'seed', 'name'],
    'tf.reduce_all': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_any': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_join': ['inputs', 'axis', 'keep_dims', 'separator', 'name', 'reduction_indices', 'keepdims'],
    'tf.reduce_logsumexp': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_max': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_mean': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_min': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_prod': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reduce_sum': ['input_tensor', 'axis', 'keepdims', 'name', 'reduction_indices', 'keep_dims'],
    'tf.reverse_sequence': ['input', 'seq_lengths', 'seq_axis', 'batch_axis', 'name', 'seq_dim', 'batch_dim'],
    'tf.serialize_many_sparse': ['sp_input', 'name', 'out_type'],
    'tf.serialize_sparse': ['sp_input', 'name', 'out_type'],
    'tf.shape': ['input', 'name', 'out_type'],
    'tf.size': ['input', 'name', 'out_type'],
    'tf.space_to_batch': ['input', 'paddings', 'block_size', 'name', 'block_shape'],
    'tf.space_to_depth': ['input', 'block_size', 'name', 'data_format'],
    'tf.sparse.add': ['a', 'b', 'threshold', 'thresh'],
    'tf.sparse.concat': ['axis', 'sp_inputs', 'name', 'expand_nonconcat_dim', 'concat_dim', 'expand_nonconcat_dims'],
    'tf.sparse.reduce_max': ['sp_input', 'axis', 'keepdims', 'reduction_axes', 'keep_dims'],
    'tf.sparse.segment_mean': ['data', 'indices', 'segment_ids', 'name', 'num_segments'],
    'tf.sparse.segment_sqrt_n': ['data', 'indices', 'segment_ids', 'name', 'num_segments'],
    'tf.sparse.segment_sum': ['data', 'indices', 'segment_ids', 'name', 'num_segments'],
    'tf.sparse.split': ['keyword_required', 'sp_input', 'num_split', 'axis', 'name', 'split_dim'],
    'tf.sparse_add': ['a', 'b', 'threshold', 'thresh'],
    'tf.sparse_concat': ['axis', 'sp_inputs', 'name', 'expand_nonconcat_dim', 'concat_dim', 'expand_nonconcat_dims'],
    'tf.sparse_matmul': ['a', 'b', 'transpose_a', 'transpose_b', 'a_is_sparse', 'b_is_sparse', 'name'],
    'tf.sparse_reduce_max': ['sp_input', 'axis', 'keepdims', 'reduction_axes', 'keep_dims'],
    'tf.sparse_segment_mean': ['data', 'indices', 'segment_ids', 'name', 'num_segments'],
    'tf.sparse_segment_sqrt_n': ['data', 'indices', 'segment_ids', 'name', 'num_segments'],
    'tf.sparse_segment_sum': ['data', 'indices', 'segment_ids', 'name', 'num_segments'],
    'tf.sparse_split': ['keyword_required', 'sp_input', 'num_split', 'axis', 'name', 'split_dim'],
    'tf.strings.length': ['input', 'name', 'unit'],
    'tf.strings.reduce_join': ['inputs', 'axis', 'keep_dims', 'separator', 'name', 'reduction_indices', 'keepdims'],
    'tf.strings.substr': ['input', 'pos', 'len', 'name', 'unit'],
    'tf.substr': ['input', 'pos', 'len', 'name', 'unit'],
    'tf.test.assert_equal_graph_def': ['actual', 'expected', 'checkpoint_v2', 'hash_table_shared_name'],
    'tf.train.sdca_fprint': ['input', 'name'],
    'tf.train.sdca_optimizer': ['sparse_example_indices', 'sparse_feature_indices', 'sparse_feature_values', 'dense_features', 'example_weights', 'example_labels', 'sparse_indices', 'sparse_weights', 'dense_weights', 'example_state_data', 'loss_type', 'l1', 'l2', 'num_loss_partitions', 'num_inner_iterations', 'adaptative', 'name'],
    'tf.train.sdca_shrink_l1': ['weights', 'l1', 'l2', 'name'],
    'tf.transpose': ['a', 'perm', 'name', 'conjugate'],
    'tf.tuple': ['tensors', 'name', 'control_inputs'],
    'tf.uniform_unit_scaling_initializer': ['factor', 'seed', 'dtype'],
    'tf.while_loop': ['cond', 'body', 'loop_vars', 'shape_invariants', 'parallel_iterations', 'back_prop', 'swap_memory', 'name', 'maximum_iterations', 'return_same_structure']
}
