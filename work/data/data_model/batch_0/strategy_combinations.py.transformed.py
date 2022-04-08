
from tensorflow.python.distribute import strategy_combinations
multidevice_strategies = [
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.tpu_strategy,
]
multiworker_strategies = [
    strategy_combinations.multi_worker_mirrored_2x1_cpu,
    strategy_combinations.multi_worker_mirrored_2x1_gpu,
    strategy_combinations.multi_worker_mirrored_2x2_gpu
]
strategies_minus_default_minus_tpu = [
    strategy_combinations.one_device_strategy,
    strategy_combinations.one_device_strategy_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.central_storage_strategy_with_gpu_and_cpu
]
strategies_minus_tpu = [
    strategy_combinations.default_strategy,
    strategy_combinations.one_device_strategy,
    strategy_combinations.one_device_strategy_gpu,
    strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
    strategy_combinations.mirrored_strategy_with_two_gpus,
    strategy_combinations.central_storage_strategy_with_gpu_and_cpu
]
multi_worker_mirrored_strategies = [
    strategy_combinations.multi_worker_mirrored_2x1_cpu,
    strategy_combinations.multi_worker_mirrored_2x1_gpu,
    strategy_combinations.multi_worker_mirrored_2x2_gpu
]
tpu_strategies = [
    strategy_combinations.tpu_strategy,
]
parameter_server_strategies_single_worker = [
    strategy_combinations.parameter_server_strategy_1worker_2ps_cpu,
    strategy_combinations.parameter_server_strategy_1worker_2ps_1gpu,
]
parameter_server_strategies_multi_worker = [
    strategy_combinations.parameter_server_strategy_3worker_2ps_cpu,
    strategy_combinations.parameter_server_strategy_3worker_2ps_1gpu,
]
all_strategies = strategies_minus_tpu + tpu_strategies
