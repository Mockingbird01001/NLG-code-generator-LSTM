
from enum import Enum
class NoValue(Enum):
    return self.name
class NodeAnno(NoValue):
  IS_LOCAL = 'Symbol is local to the function scope being analyzed.'
  IS_PARAM = 'Symbol is a parameter to the function being analyzed.'
  IS_MODIFIED_SINCE_ENTRY = (
      'Symbol has been explicitly replaced in the current function scope.')
  ARGS_SCOPE = 'The scope for the argument list of a function call.'
  COND_SCOPE = 'The scope for the test node of a conditional statement.'
  ITERATE_SCOPE = 'The scope for the iterate assignment of a for loop.'
  ARGS_AND_BODY_SCOPE = (
      'The scope for the main body of a function or lambda, including its'
      ' arguments.')
  BODY_SCOPE = (
      'The scope for the main body of a statement (True branch for if '
      'statements, main body for loops).')
  ORELSE_SCOPE = (
      'The scope for the orelse body of a statement (False branch for if '
      'statements, orelse body for loops).')
