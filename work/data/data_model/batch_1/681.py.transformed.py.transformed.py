
import os
from pip._vendor.pep517.wrappers import Pep517HookCaller
from pip._internal.build_env import BuildEnvironment
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory
def generate_metadata(build_env, backend):
    metadata_tmpdir = TempDirectory(
        kind="modern-metadata", globally_managed=True
    )
    metadata_dir = metadata_tmpdir.path
    with build_env:
        runner = runner_with_spinner_message("Preparing wheel metadata")
        with backend.subprocess_runner(runner):
            distinfo_dir = backend.prepare_metadata_for_build_wheel(
                metadata_dir
            )
    return os.path.join(metadata_dir, distinfo_dir)
