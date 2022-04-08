
import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import lit.util
config.name = 'MLIR_HLO_OPT'
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir', '.mlir.py']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mlir_hlo_obj_root, 'test')
config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()
config.excludes = [
    'Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt'
]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mlir_hlo_obj_root, 'tests')
config.mlir_hlo_tools_dir = os.path.join(config.mlir_hlo_obj_root, 'bin')
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
tool_dirs = [
    config.mlir_hlo_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    'mlir-hlo-opt',
    'mlir-cpu-runner',
    ToolSubst(
        '%mlir_runner_utils_dir',
        config.mlir_runner_utils_dir,
        unresolved='ignore'),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
