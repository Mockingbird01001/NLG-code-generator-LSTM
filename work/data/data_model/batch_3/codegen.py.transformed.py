
import random
import string
import gast
import numpy as np
from tensorflow.python.autograph.pyct import templates
class NodeSampler(object):
  sample_map = None
  def sample(self):
    nodes, magnitudes = zip(*self.sample_map.items())
    return np.random.choice(
        nodes, p=np.array(magnitudes, dtype='float32') / np.sum(magnitudes))
class StatementSampler(NodeSampler):
  sample_map = dict((
      (gast.Assign, 10),
      (gast.Print, 1),
      (gast.If, 2),
      (gast.While, 2),
      (gast.For, 0),
  ))
class ExpressionSampler(NodeSampler):
  sample_map = dict((
      (gast.UnaryOp, 1),
      (gast.BinOp, 8),
      (gast.Name, 1),
      (gast.Call, 0),
  ))
class CompareSampler(NodeSampler):
  sample_map = dict((
      (gast.Eq, 1),
      (gast.NotEq, 1),
      (gast.Lt, 1),
      (gast.LtE, 1),
      (gast.Gt, 1),
      (gast.GtE, 1),
      (gast.Is, 1),
      (gast.IsNot, 1),
  ))
class BinaryOpSampler(NodeSampler):
  sample_map = dict((
      (gast.Add, 1),
      (gast.Sub, 1),
      (gast.Mult, 1),
      (gast.Div, 1),
      (gast.FloorDiv, 1),
      (gast.Mod, 1),
      (gast.Pow, 1),
  ))
class UnaryOpSampler(NodeSampler):
  sample_map = dict(((gast.USub, 1), (gast.UAdd, 0)))
class NameSampler(NodeSampler):
  sample_map = dict((
      ('new', 1),
      ('existing', 1),
  ))
N_CONTROLFLOW_STATEMENTS = 10
N_FUNCTIONDEF_STATEMENTS = 10
class CodeGenerator(object):
  def __init__(self, max_depth=3, depth=0):
    self.max_depth = max_depth
    self.depth = depth
  def generate_statement(self):
    desired_node = StatementSampler().sample()
    self.depth += 1
    if desired_node in (gast.While, gast.For, gast.If):
      if self.depth > self.max_depth:
        return self.generate_statement()
    method = 'generate_' + desired_node.__name__
    visitor = getattr(self, method)
    node = visitor()
    self.depth -= 1
    return node
  def sample_node_list(self, low, high, generator):
    statements = []
    for _ in range(np.random.randint(low, high)):
      statements.append(generator())
    return statements
  def generate_Name(self, ctx=gast.Load()):
    variable_name = '_' + ''.join(
        random.choice(string.ascii_lowercase) for _ in range(4))
    return gast.Name(variable_name, ctx=ctx, annotation=None)
  def generate_BinOp(self):
    op = BinaryOpSampler().sample()()
    return gast.BinOp(self.generate_Name(), op, self.generate_Name())
  def generate_Compare(self):
    op = CompareSampler().sample()()
    return gast.Compare(self.generate_Name(), [op], [self.generate_Name()])
  def generate_UnaryOp(self):
    operand = self.generate_Name()
    op = UnaryOpSampler().sample()()
    return gast.UnaryOp(op, operand)
  def generate_expression(self):
    desired_node = ExpressionSampler().sample()
    method = 'generate_' + desired_node.__name__
    generator = getattr(self, method)
    return generator()
  def generate_Assign(self):
    target_node = self.generate_Name(gast.Store())
    value_node = self.generate_expression()
    node = gast.Assign(targets=[target_node], value=value_node)
    return node
  def generate_If(self):
    test = self.generate_Compare()
    body = self.sample_node_list(
        low=1,
        high=N_CONTROLFLOW_STATEMENTS // 2,
        generator=self.generate_statement)
    orelse = self.sample_node_list(
        low=1,
        high=N_CONTROLFLOW_STATEMENTS // 2,
        generator=self.generate_statement)
    node = gast.If(test, body, orelse)
    return node
  def generate_While(self):
    test = self.generate_Compare()
    body = self.sample_node_list(
        low=1, high=N_CONTROLFLOW_STATEMENTS, generator=self.generate_statement)
    node = gast.While(test, body, orelse)
    return node
  def generate_Call(self):
    raise NotImplementedError
  def generate_Return(self):
    return gast.Return(self.generate_expression())
  def generate_Print(self):
    return templates.replace('print(x)', x=self.generate_expression())[0]
  def generate_FunctionDef(self):
    arg_vars = self.sample_node_list(
        low=2, high=10, generator=lambda: self.generate_Name(gast.Param()))
    args = gast.arguments(arg_vars, None, [], [], None, [])
    body = self.sample_node_list(
        low=1, high=N_FUNCTIONDEF_STATEMENTS, generator=self.generate_statement)
    body.append(self.generate_Return())
    fn_name = self.generate_Name().id
    node = gast.FunctionDef(fn_name, args, body, (), None)
    return node
def generate_random_functiondef():
  return CodeGenerator().generate_FunctionDef()
