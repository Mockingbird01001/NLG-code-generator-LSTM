
import re
import sys
import six
from six.moves import range
import six.moves.configparser
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
PATH_TO_DIR = "tensorflow/tools/tensorflow_builder/compat_checker"
def _compare_versions(v1, v2):
  if v1 == "inf" and v2 == "inf":
    raise RuntimeError("Cannot compare `inf` to `inf`.")
  rtn_dict = {"smaller": None, "larger": None}
  v1_list = six.ensure_str(v1).split(".")
  v2_list = six.ensure_str(v2).split(".")
  if v1_list[0] == "inf":
    v1_list[0] = str(int(v2_list[0]) + 1)
  if v2_list[0] == "inf":
    v2_list[0] = str(int(v1_list[0]) + 1)
  v_long = v1_list if len(v1_list) >= len(v2_list) else v2_list
  v_short = v1_list if len(v1_list) < len(v2_list) else v2_list
  larger, smaller = None, None
  for i, ver in enumerate(v_short, start=0):
    if int(ver) > int(v_long[i]):
      larger = _list_to_string(v_short, ".")
      smaller = _list_to_string(v_long, ".")
    elif int(ver) < int(v_long[i]):
      larger = _list_to_string(v_long, ".")
      smaller = _list_to_string(v_short, ".")
    else:
      if i == len(v_short) - 1:
        if v_long[i + 1:] == ["0"]*(len(v_long) - 1 - i):
          larger = "equal"
          smaller = "equal"
        else:
          larger = _list_to_string(v_long, ".")
          smaller = _list_to_string(v_short, ".")
      else:
        pass
    if larger:
      break
  rtn_dict["smaller"] = smaller
  rtn_dict["larger"] = larger
  return rtn_dict
def _list_to_string(l, s):
  return s.join(l)
def _get_func_name():
  return tf_inspect.stack()[1][3]
class ConfigCompatChecker(object):
  class _Reqs(object):
    """Class that stores specifications related to a single requirement.
    `_Reqs` represents a single version or dependency requirement specified in
    the `.ini` config file. It is meant ot be used inside `ConfigCompatChecker`
    to help organize and identify version and dependency compatibility for a
    given configuration (e.g. gcc version) required by the client.
    """
    def __init__(self, req, config, section):
      """Initializes a version or dependency requirement object.
      Args:
        req: List that contains individual supported versions or a single string
             that contains `range` definition.
               e.g. [`range(1.0, 2.0) include(3.0) exclude(1.5)`]
               e.g. [`1.0`, `3.0`, `7.1`]
        config: String that is the configuration name.
                  e.g. `platform`
        section: String that is the section name from the `.ini` config file
                 under which the requirement is defined.
                   e.g. `Required`, `Optional`, `Unsupported`, `Dependency`
      """
      self.req = req
      self.exclude = None
      self.include = None
      self.config = config
      self._section = section
      self._initialized = None
      self._error_message = []
      self.parse_single_req()
    @property
    def get_status(self):
      """Get status of `_Reqs` initialization.
      Returns:
        Tuple
          (Boolean indicating initialization status,
           List of error messages, if any)
      """
      return self._initialized, self._error_message
    def __str__(self):
      info = {
          "section": self._section,
          "config": self.config,
          "req_type": self._req_type,
          "req": str(self.req),
          "range": str(self.range),
          "exclude": str(self.exclude),
          "include": str(self.include),
          "init": str(self._initialized)
      }
      req_str = "\n >>> _Reqs Instance <<<\n"
      req_str += "Section: {section}\n"
      req_str += "Configuration name: {config}\n"
      req_str += "Requirement type: {req_type}\n"
      req_str += "Requirement: {req}\n"
      req_str += "Range: {range}\n"
      req_str += "Exclude: {exclude}\n"
      req_str += "Include: {include}\n"
      req_str += "Initialized: {init}\n\n"
      return req_str.format(**info)
    def parse_single_req(self):
      """Parses a requirement and stores information.
      `self.req` _initialized in `__init__` is called for retrieving the
      requirement.
      A requirement can come in two forms:
        [1] String that includes `range` indicating range syntax for defining
            a requirement.
              e.g. `range(1.0, 2.0) include(3.0) exclude(1.5)`
        [2] List that includes individual supported versions or items.
              e.g. [`1.0`, `3.0`, `7.1`]
      For a list type requirement, it directly stores the list to
      `self.include`.
      Call `get_status` for checking the status of the parsing. This function
      sets `self._initialized` to `False` and immediately returns with an error
      message upon encountering a failure. It sets `self._initialized` to `True`
      and returns without an error message upon success.
      """
      expr = r"(range\()?([\d\.\,\s]+)(\))?( )?(include\()?"
      expr += r"([\d\.\,\s]+)?(\))?( )?(exclude\()?([\d\.\,\s]+)?(\))?"
      if not self.req:
        err_msg = "[Error] Requirement is missing. "
        err_msg += "(section = %s, " % str(self._section)
        err_msg += "config = %s, req = %s)" % (str(self.config), str(self.req))
        logging.error(err_msg)
        self._initialized = False
        self._error_message.append(err_msg)
        return
      if "range" in self.req[0]:
        self._req_type = "range"
        match = re.match(expr, self.req[0])
        if not match:
          err_msg = "[Error] Encountered issue when parsing the requirement."
          err_msg += " (req = %s, match = %s)" % (str(self.req), str(match))
          logging.error(err_msg)
          self._initialized = False
          self._error_message.append(err_msg)
          return
        else:
          match_grp = match.groups()
          match_size = len(match_grp)
          for i, m in enumerate(match_grp[0:match_size-1], start=0):
            next_match = match_grp[i + 1]
            if m not in ["", None, " ", ")"]:
              if "range" in m:
                comma_count = next_match.count(",")
                if comma_count > 1 or comma_count == 0:
                  err_msg = "[Error] Found zero or more than one comma in range"
                  err_msg += " definition. (req = %s, " % str(self.req)
                  err_msg += "match = %s)" % str(next_match)
                  logging.error(err_msg)
                  self._initialized = False
                  self._error_message.append(err_msg)
                  return
                min_max = next_match.replace(" ", "").split(",")
                if not min_max[0]:
                  min_max[0] = "0"
                if not min_max[1]:
                  min_max[1] = "inf"
                self.range = min_max
              if "exclude" in m:
                self.exclude = next_match.replace(" ", "").split(",")
              if "include" in m:
                self.include = next_match.replace(" ", "").split(",")
              self._initialized = True
      else:
        self._req_type = "no_range"
        if not isinstance(self.req, list):
          err_msg = "[Error] Requirement is not a list."
          err_msg += "(req = %s, " % str(self.req)
          err_msg += "type(req) = %s)" % str(type(self.req))
          logging.error(err_msg)
          self._initialized = False
          self._error_message.append(err_msg)
        else:
          self.include = self.req
          self._initialized = True
      return
  def __init__(self, usr_config, req_file):
    """Initializes a configuration compatibility checker.
    Args:
      usr_config: Dict of all configuration(s) whose version compatibilities are
                  to be checked against the rules defined in the `.ini` config
                  file.
      req_file: String that is the full name of the `.ini` config file.
                  e.g. `config.ini`
    """
    self.usr_config = usr_config
    self.req_file = req_file
    self.warning_msg = []
    self.error_msg = []
    reqs_all = self.get_all_reqs()
    self.required = reqs_all["required"]
    self.optional = reqs_all["optional"]
    self.unsupported = reqs_all["unsupported"]
    self.dependency = reqs_all["dependency"]
    self.successes = []
    self.failures = []
  def get_all_reqs(self):
    """Parses all compatibility specifications listed in the `.ini` config file.
    Reads and parses each and all compatibility specifications from the `.ini`
    config file by sections. It then populates appropriate dicts that represent
    each section (e.g. `self.required`) and returns a tuple of the populated
    dicts.
    Returns:
      Dict of dict
        { `required`: Dict of `Required` configs and supported versions,
          `optional`: Dict of `Optional` configs and supported versions,
          `unsupported`: Dict of `Unsupported` configs and supported versions,
          `dependency`: Dict of `Dependency` configs and supported versions }
    """
    try:
      open(self.req_file, "rb")
    except IOError:
      msg = "[Error] Cannot read file '%s'." % self.req_file
      logging.error(msg)
      sys.exit(1)
    curr_status = True
    parser = six.moves.configparser.ConfigParser()
    parser.read(self.req_file)
    if not parser.sections():
      err_msg = "[Error] Empty config file. "
      err_msg += "(file = %s, " % str(self.req_file)
      err_msg += "parser sectons = %s)" % str(parser.sections())
      self.error_msg.append(err_msg)
      logging.error(err_msg)
      curr_status = False
    required_dict = {}
    optional_dict = {}
    unsupported_dict = {}
    dependency_dict = {}
    for section in parser.sections():
      all_configs = parser.options(section)
      for config in all_configs:
        spec = parser.get(section, config)
        if section == "Dependency":
          dependency_dict[config] = []
          spec_split = spec.split(",\n")
          if spec_split[0] == "[":
            spec_split = spec_split[1:]
          elif "[" in spec_split[0]:
            spec_split[0] = spec_split[0].replace("[", "")
          else:
            warn_msg = "[Warning] Config file format error: Missing `[`."
            warn_msg += "(section = %s, " % str(section)
            warn_msg += "config = %s)" % str(config)
            logging.warning(warn_msg)
            self.warning_msg.append(warn_msg)
          if spec_split[-1] == "]":
            spec_split = spec_split[:-1]
          elif "]" in spec_split[-1]:
            spec_split[-1] = spec_split[-1].replace("]", "")
          else:
            warn_msg = "[Warning] Config file format error: Missing `]`."
            warn_msg += "(section = %s, " % str(section)
            warn_msg += "config = %s)" % str(config)
            logging.warning(warn_msg)
            self.warning_msg.append(warn_msg)
          for rule in spec_split:
            spec_dict = self.filter_dependency(rule)
            cfg_req = self._Reqs(
                self.convert_to_list(spec_dict["cfg_spec"], " "),
                config=cfg_name,
                section=section
            )
            dep_req = self._Reqs(
                self.convert_to_list(spec_dict["cfgd_spec"], " "),
                config=dep_name,
                section=section
            )
            cfg_req_status = cfg_req.get_status
            dep_req_status = dep_req.get_status
            if not cfg_req_status[0] or not dep_req_status[0]:
              msg = "[Error] Failed to create _Reqs() instance for a "
              msg += "dependency item. (config = %s, " % str(cfg_name)
              msg += "dep = %s)" % str(dep_name)
              logging.error(msg)
              self.error_msg.append(cfg_req_status[1])
              self.error_msg.append(dep_req_status[1])
              curr_status = False
              break
            else:
              dependency_dict[config].append(
                  [cfg_name, cfg_req, dep_name, dep_req])
          if not curr_status:
            break
        else:
          if section == "Required":
            add_to = required_dict
          elif section == "Optional":
            add_to = optional_dict
          elif section == "Unsupported":
            add_to = unsupported_dict
          else:
            msg = "[Error] Section name `%s` is not accepted." % str(section)
            msg += "Accepted section names are `Required`, `Optional`, "
            msg += "`Unsupported`, and `Dependency`."
            logging.error(msg)
            self.error_msg.append(msg)
            curr_status = False
            break
          req_list = self.convert_to_list(self.filter_line(spec), " ")
          add_to[config] = self._Reqs(req_list, config=config, section=section)
        if not curr_status:
          break
      if not curr_status:
        break
    return_dict = {
        "required": required_dict,
        "optional": optional_dict,
        "unsupported": unsupported_dict,
        "dependency": dependency_dict
    }
    return return_dict
  def filter_dependency(self, line):
    """Filters dependency compatibility rules defined in the `.ini` config file.
    Dependency specifications are defined as the following:
      `<config> <config_version> requires <dependency> <dependency_version>`
    e.g.
      `python 3.7 requires tensorflow 1.13`
      `tensorflow range(1.0.0, 1.13.1) requires gcc range(4.8, )`
    Args:
      line: String that is a dependency specification defined under `Dependency`
            section in the `.ini` config file.
    Returns:
      Dict with configuration and its dependency information.
    """
    line = line.strip("\n")
    expr = r"(?P<cfg>[\S]+) (?P<cfg_spec>range\([\d\.\,\s]+\)( )?"
    expr += r"(include\([\d\.\,\s]+\))?( )?(exclude\([\d\.\,\s]+\))?( )?"
    expr += r"|[\d\,\.\s]+) requires (?P<cfgd>[\S]+) (?P<cfgd_spec>range"
    expr += r"\([\d\.\,\s]+\)( )?(include\([\d\.\,\s]+\))?( )?"
    expr += r"(exclude\([\d\.\,\s]+\))?( )?|[\d\,\.\s]+)"
    r = re.match(expr, line.strip("\n"))
    return r.groupdict()
  def convert_to_list(self, item, separator):
    out = None
    if not isinstance(item, list):
      if "range" in item:
        out = [item]
      else:
        out = item.split(separator)
        for i in range(len(out)):
          out[i] = out[i].replace(",", "")
    else:
      out = [item]
    return out
  def filter_line(self, line):
    filtered = []
    warn_msg = []
    splited = line.split("\n")
    if not line and len(splited) < 1:
      warn_msg = "[Warning] Empty line detected while filtering lines."
      logging.warning(warn_msg)
      self.warning_msg.append(warn_msg)
    if splited[0] == "[":
      filtered = splited[1:]
    elif "[" in splited[0]:
      splited = splited[0].replace("[", "")
      filtered = splited
    else:
      warn_msg = "[Warning] Format error. `[` could be missing in "
      warn_msg += "the config (.ini) file. (line = %s)" % str(line)
      logging.warning(warn_msg)
      self.warning_msg.append(warn_msg)
    if filtered[-1] == "]":
      filtered = filtered[:-1]
    elif "]" in filtered[-1]:
      filtered[-1] = six.ensure_str(filtered[-1]).replace("]", "")
    else:
      warn_msg = "[Warning] Format error. `]` could be missing in "
      warn_msg += "the config (.ini) file. (line = %s)" % str(line)
      logging.warning(warn_msg)
      self.warning_msg.append(warn_msg)
    return filtered
  def in_range(self, ver, req):
    """Checks if a version satisfies a version and/or compatibility requirement.
    Args:
      ver: List whose first item is a config version that needs to be checked
           for support status and version compatibility.
             e.g. ver = [`1.0`]
      req: `_Reqs` class instance that represents a configuration version and
            compatibility specifications.
    Returns:
      Boolean output of checking if version `ver` meets the requirement
        stored in `req` (or a `_Reqs` requirements class instance).
    """
    if req.exclude is not None:
      for v in ver:
        if v in req.exclude:
          return False
    include_checked = False
    if req.include is not None:
      for v in ver:
        if v in req.include:
          return True
      include_checked = True
    if req.range != [None, None]:
      if lg in [ver, "equal"] and sm in [ver, "equal", "inf"]:
        return True
      else:
        err_msg = "[Error] Version is outside of supported range. "
        err_msg += "(config = %s, " % str(req.config)
        err_msg += "version = %s, " % str(ver)
        err_msg += "supported range = %s)" % str(req.range)
        logging.warning(err_msg)
        self.warning_msg.append(err_msg)
        return False
    else:
      err_msg = ""
      if include_checked:
        err_msg = "[Error] Version is outside of supported range. "
      else:
        err_msg = "[Error] Missing specification. "
      err_msg += "(config = %s, " % str(req.config)
      err_msg += "version = %s, " % str(ver)
      err_msg += "supported range = %s)" % str(req.range)
      logging.warning(err_msg)
      self.warning_msg.append(err_msg)
      return False
  def _print(self, *args):
    """Prints compatibility check status and failure or warning messages.
    Prints to console without using `logging`.
    Args:
      *args: String(s) that is one of:
    Raises:
      Exception: If *args not in:
                   [`failures`, `successes`, `failure_msgs`, `warning_msg`]
    """
    def _format(name, arr):
      tlen = len(title)
      print("-"*tlen)
      print(title)
      print("-"*tlen)
      if arr:
        for item in arr:
          detail = ""
          if isinstance(item[1], list):
            for itm in item[1]:
              detail += str(itm) + ", "
            detail = detail[:-2]
          else:
            detail = str(item[1])
          print("  %s ('%s')\n" % (str(item[0]), detail))
      else:
        print("  No %s" % name)
      print("\n")
    for p_item in args:
      if p_item == "failures":
        _format("Failures", self.failures)
      elif p_item == "successes":
        _format("Successes", self.successes)
      elif p_item == "failure_msgs":
        _format("Failure Messages", self.error_msg)
      elif p_item == "warning_msgs":
        _format("Warning Messages", self.warning_msg)
      else:
        raise Exception(
            "[Error] Wrong input provided for %s." % _get_func_name())
  def check_compatibility(self):
    """Checks version and dependency compatibility for a given configuration.
    `check_compatibility` immediately returns with `False` (or failure status)
    if any child process or checks fail. For error and warning messages, either
    print `self.(error_msg|warning_msg)` or call `_print` function.
    Returns:
      Boolean that is a status of the compatibility check result.
    """
    usr_keys = list(self.usr_config.keys())
    for k in six.iterkeys(self.usr_config):
      if k not in usr_keys:
        err_msg = "[Error] Required config not found in user config."
        err_msg += "(required = %s, " % str(k)
        err_msg += "user configs = %s)" % str(usr_keys)
        logging.error(err_msg)
        self.error_msg.append(err_msg)
        self.failures.append([k, err_msg])
        return False
    overall_status = True
    for config_name, spec in six.iteritems(self.usr_config):
      temp_status = True
      in_required = config_name in list(self.required.keys())
      in_optional = config_name in list(self.optional.keys())
      in_unsupported = config_name in list(self.unsupported.keys())
      in_dependency = config_name in list(self.dependency.keys())
      if not (in_required or in_optional or in_unsupported or in_dependency):
        warn_msg = "[Error] User config not defined in config file."
        warn_msg += "(user config = %s)" % str(config_name)
        logging.warning(warn_msg)
        self.warning_msg.append(warn_msg)
        self.failures.append([config_name, warn_msg])
        temp_status = False
      else:
        if in_unsupported:
          if self.in_range(spec, self.unsupported[config_name]):
            err_msg = "[Error] User config is unsupported. It is "
            err_msg += "defined under 'Unsupported' section in the config file."
            err_msg += " (config = %s, spec = %s)" % (config_name, str(spec))
            logging.error(err_msg)
            self.error_msg.append(err_msg)
            self.failures.append([config_name, err_msg])
            temp_status = False
        if in_required:
          if not self.in_range(spec, self.required[config_name]):
            err_msg = "[Error] User config cannot be supported. It is not in "
            err_msg += "the supported range as defined in the 'Required' "
            err_msg += "section. (config = %s, " % config_name
            err_msg += "spec = %s)" % str(spec)
            logging.error(err_msg)
            self.error_msg.append(err_msg)
            self.failures.append([config_name, err_msg])
            temp_status = False
        if in_optional:
          if not self.in_range(spec, self.optional[config_name]):
            err_msg = "[Error] User config cannot be supported. It is not in "
            err_msg += "the supported range as defined in the 'Optional' "
            err_msg += "section. (config = %s, " % config_name
            err_msg += "spec = %s)" % str(spec)
            logging.error(err_msg)
            self.error_msg.append(err_msg)
            self.failures.append([config_name, err_msg])
            temp_status = False
        if in_dependency:
          dep_list = self.dependency[config_name]
          if dep_list:
            for rule in dep_list:
              try:
                cfg_name = self.usr_config[cfg]
                dep_name = self.usr_config[dep]
                cfg_status = self.in_range(cfg_name, cfg_req)
                dep_status = self.in_range(dep_name, dep_req)
                if cfg_status:
                  if not dep_status:
                    err_msg = "[Error] User config has a dependency that cannot"
                    err_msg += " be supported. "
                    err_msg += "'%s' has a dependency on " % str(config_name)
                    err_msg += "'%s'." % str(dep)
                    logging.error(err_msg)
                    self.error_msg.append(err_msg)
                    self.failures.append([config_name, err_msg])
                    temp_status = False
              except KeyError:
                err_msg = "[Error] Dependency is missing from `Required`. "
                err_msg += "(config = %s, ""dep = %s)" % (cfg, dep)
                logging.error(err_msg)
                self.error_msg.append(err_msg)
                self.failures.append([config_name, err_msg])
                temp_status = False
      if temp_status:
        self.successes.append([config_name, spec])
      else:
        overall_status = False
    return overall_status
