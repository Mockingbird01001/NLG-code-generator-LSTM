
"""Prints ROCm library and header directories and versions found on the system.
The script searches for ROCm library and header files on the system, inspects
them to determine their version and prints the configuration to stdout.
The path to inspect is specified through an environment variable (ROCM_PATH).
If no valid configuration is found, the script prints to stderr and
returns an error code.
The script takes the directory specified by the ROCM_PATH environment variable.
The script looks for headers and library files in a hard-coded set of
subdirectories from base path of the specified directory. If ROCM_PATH is not
specified, then "/opt/rocm" is used as it default value
"""
import io
import os
import re
import sys
class ConfigError(Exception):
  pass
def _get_default_rocm_path():
  return "/opt/rocm"
def _get_rocm_install_path():
  rocm_install_path = _get_default_rocm_path()
  if "ROCM_PATH" in os.environ:
    rocm_install_path = os.environ["ROCM_PATH"]
  return rocm_install_path
def _get_composite_version_number(major, minor, patch):
  return 10000 * major + 100 * minor + patch
def _get_header_version(path, name):
  for line in io.open(path, "r", encoding="utf-8"):
    if match:
      value = match.group(1)
      return int(value)
                    "  not present in file {} OR\n".format(path) +
                    "  its value is not an integer literal")
def _find_rocm_config(rocm_install_path):
  def rocm_version_numbers_pre_rocm50(path, prior_err):
    version_file = os.path.join(path, ".info/version-dev")
    if not os.path.exists(version_file):
      raise ConfigError("{} ROCm version file ".format(prior_err) +
                        '"{}" not found either.'.format(version_file))
    version_numbers = []
    with open(version_file) as f:
      version_string = f.read().strip()
      version_numbers = version_string.split(".")
    major = int(version_numbers[0])
    minor = int(version_numbers[1])
    patch = int(version_numbers[2].split("-")[0])
    return major, minor, patch
  def rocm_version_numbers_post_rocm50(path):
    version_file = os.path.join(path, "include/rocm_version.h")
    if not os.path.exists(version_file):
      return False, 'ROCm version file "{}" not found.'.format(version_file) +\
        " Trying an alternate approach to determine the ROCm version.", 0,0,0
    major = _get_header_version(version_file, "ROCM_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCM_VERSION_MINOR")
    patch = _get_header_version(version_file, "ROCM_VERSION_PATCH")
    return True, "", major, minor, patch
  status, error_msg, major, minor, patch = \
    rocm_version_numbers_post_rocm50(rocm_install_path)
  if not status:
    major, minor, patch = \
      rocm_version_numbers_pre_rocm50(rocm_install_path, error_msg)
  rocm_config = {
      "rocm_version_number": _get_composite_version_number(major, minor, patch)
  }
  return rocm_config
def _find_hipruntime_config(rocm_install_path):
  def hipruntime_version_number(path):
    version_file = os.path.join(path, "hip/include/hip/hip_version.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'HIP Runtime version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "HIP_VERSION_MAJOR")
    minor = _get_header_version(version_file, "HIP_VERSION_MINOR")
    return 100 * major + minor
  hipruntime_config = {
      "hipruntime_version_number": hipruntime_version_number(rocm_install_path)
  }
  return hipruntime_config
def _find_miopen_config(rocm_install_path):
  def miopen_version_numbers(path):
    version_file = os.path.join(path, "miopen/include/miopen/version.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'MIOpen version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "MIOPEN_VERSION_MAJOR")
    minor = _get_header_version(version_file, "MIOPEN_VERSION_MINOR")
    patch = _get_header_version(version_file, "MIOPEN_VERSION_PATCH")
    return major, minor, patch
  major, minor, patch = miopen_version_numbers(rocm_install_path)
  miopen_config = {
      "miopen_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return miopen_config
def _find_rocblas_config(rocm_install_path):
  def rocblas_version_numbers(path):
    possible_version_files = [
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "rocblas version file not found in {}".format(
              possible_version_files))
    major = _get_header_version(version_file, "ROCBLAS_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCBLAS_VERSION_MINOR")
    patch = _get_header_version(version_file, "ROCBLAS_VERSION_PATCH")
    return major, minor, patch
  major, minor, patch = rocblas_version_numbers(rocm_install_path)
  rocblas_config = {
      "rocblas_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return rocblas_config
def _find_rocrand_config(rocm_install_path):
  def rocrand_version_number(path):
    possible_version_files = [
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError(
          "rocrand version file not found in {}".format(possible_version_files))
    version_number = _get_header_version(version_file, "ROCRAND_VERSION")
    return version_number
  rocrand_config = {
      "rocrand_version_number": rocrand_version_number(rocm_install_path)
  }
  return rocrand_config
def _find_rocfft_config(rocm_install_path):
  def rocfft_version_numbers(path):
    version_file = os.path.join(path, "rocfft/include/rocfft-version.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'rocfft version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "rocfft_version_major")
    minor = _get_header_version(version_file, "rocfft_version_minor")
    patch = _get_header_version(version_file, "rocfft_version_patch")
    return major, minor, patch
  major, minor, patch = rocfft_version_numbers(rocm_install_path)
  rocfft_config = {
      "rocfft_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return rocfft_config
def _find_hipfft_config(rocm_install_path):
  def hipfft_version_numbers(path):
    version_file = os.path.join(path, "hipfft/include/hipfft-version.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'hipfft version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "hipfftVersionMajor")
    minor = _get_header_version(version_file, "hipfftVersionMinor")
    patch = _get_header_version(version_file, "hipfftVersionPatch")
    return major, minor, patch
  major, minor, patch = hipfft_version_numbers(rocm_install_path)
  hipfft_config = {
      "hipfft_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return hipfft_config
def _find_roctracer_config(rocm_install_path):
  def roctracer_version_numbers(path):
    version_file = os.path.join(path, "roctracer/include/roctracer.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'roctracer version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "ROCTRACER_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCTRACER_VERSION_MINOR")
    patch = 0
    return major, minor, patch
  major, minor, patch = roctracer_version_numbers(rocm_install_path)
  roctracer_config = {
      "roctracer_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return roctracer_config
def _find_hipsparse_config(rocm_install_path):
  def hipsparse_version_numbers(path):
    version_file = os.path.join(path, "hipsparse/include/hipsparse-version.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'hipsparse version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "hipsparseVersionMajor")
    minor = _get_header_version(version_file, "hipsparseVersionMinor")
    patch = _get_header_version(version_file, "hipsparseVersionPatch")
    return major, minor, patch
  major, minor, patch = hipsparse_version_numbers(rocm_install_path)
  hipsparse_config = {
      "hipsparse_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return hipsparse_config
def _find_hipsolver_config(rocm_install_path):
  def hipsolver_version_numbers(path):
    possible_version_files = [
    ]
    version_file = None
    for f in possible_version_files:
      version_file_path = os.path.join(path, f)
      if os.path.exists(version_file_path):
        version_file = version_file_path
        break
    if not version_file:
      raise ConfigError("hipsolver version file not found in {}".format(
          possible_version_files))
    major = _get_header_version(version_file, "hipsolverVersionMajor")
    minor = _get_header_version(version_file, "hipsolverVersionMinor")
    patch = _get_header_version(version_file, "hipsolverVersionPatch")
    return major, minor, patch
  major, minor, patch = hipsolver_version_numbers(rocm_install_path)
  hipsolver_config = {
      "hipsolver_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return hipsolver_config
def _find_rocsolver_config(rocm_install_path):
  def rocsolver_version_numbers(path):
    version_file = os.path.join(path, "rocsolver/include/rocsolver-version.h")
    if not os.path.exists(version_file):
      raise ConfigError(
          'rocsolver version file "{}" not found'.format(version_file))
    major = _get_header_version(version_file, "ROCSOLVER_VERSION_MAJOR")
    minor = _get_header_version(version_file, "ROCSOLVER_VERSION_MINOR")
    patch = _get_header_version(version_file, "ROCSOLVER_VERSION_PATCH")
    return major, minor, patch
  major, minor, patch = rocsolver_version_numbers(rocm_install_path)
  rocsolver_config = {
      "rocsolver_version_number":
          _get_composite_version_number(major, minor, patch)
  }
  return rocsolver_config
def find_rocm_config():
  rocm_install_path = _get_rocm_install_path()
  if not os.path.exists(rocm_install_path):
    raise ConfigError(
        'Specified ROCM_PATH "{}" does not exist'.format(rocm_install_path))
  result = {}
  result["rocm_toolkit_path"] = rocm_install_path
  result.update(_find_rocm_config(rocm_install_path))
  result.update(_find_hipruntime_config(rocm_install_path))
  result.update(_find_miopen_config(rocm_install_path))
  result.update(_find_rocblas_config(rocm_install_path))
  result.update(_find_rocrand_config(rocm_install_path))
  result.update(_find_rocfft_config(rocm_install_path))
  if result["rocm_version_number"] >= 40100:
    result.update(_find_hipfft_config(rocm_install_path))
  result.update(_find_roctracer_config(rocm_install_path))
  result.update(_find_hipsparse_config(rocm_install_path))
  if result["rocm_version_number"] >= 40500:
    result.update(_find_hipsolver_config(rocm_install_path))
  result.update(_find_rocsolver_config(rocm_install_path))
  return result
def main():
  try:
    for key, value in sorted(find_rocm_config().items()):
      print("%s: %s" % (key, value))
  except ConfigError as e:
    sys.stderr.write("\nERROR: {}\n\n".format(str(e)))
    sys.exit(1)
if __name__ == "__main__":
  main()