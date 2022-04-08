
from tensorflow.python.framework import ops
@ops.RegisterGradient("RiscAbs")
def _RiscAbsGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscAdd")
def _RiscAddGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscBinaryArithmetic")
def _RiscBinaryArithmeticGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscBinaryComparison")
def _RiscBinaryComparisonGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscBitcast")
def _RiscBitcastGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscBroadcast")
def _RiscBroadcastGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscCast")
def _RiscCastGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscCholesky")
def _RiscCholeskyGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscCeil")
def _RiscCeilGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscConcat")
def _RiscConcatGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscCondition")
def _RiscConditionGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscConv")
def _RiscConvGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscCos")
def _RiscCosGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscDiv")
def _RiscDivGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscDot")
def _RiscDotGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscExp")
def _RiscExpGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscFft")
def _RiscFftGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscFloor")
def _RiscFloorGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscGather")
def _RiscGatherGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscImag")
def _RiscImagGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscIsFinite")
def _RiscIsFiniteGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscLog")
def _RiscLogGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscLogicalAnd")
def _RiscLogicalAndGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscLogicalNot")
def _RiscLogicalNotGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscLogicalOr")
def _RiscLogicalOrGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscMax")
def _RiscMaxGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscMin")
def _RiscMinGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscMul")
def _RiscMulGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscNeg")
def _RiscNegGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscPad")
def _RiscPadGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscPool")
def _RiscPoolGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscPow")
def _RiscPowGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscRandomUniform")
def _RiscRandomUniformGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscReal")
def _RiscRealGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscReduce")
def _RiscReduceGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscRem")
def _RiscRemGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscReshape")
def _RiscReshapeGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscReverse")
def _RiscReverseGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscScatter")
def _RiscScatterGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscShape")
def _RiscShapeGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscSign")
def _RiscSignGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscSlice")
def _RiscSliceGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscSort")
def _RiscSortGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscSqueeze")
def _RiscSqueezeGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscSub")
def _RiscSubGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscTranspose")
def _RiscTransposeGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscTriangularSolve")
def _RiscTriangularSolvesGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscUnary")
def _RiscUnaryGrad(_, grad):
  return None, None
@ops.RegisterGradient("RiscWhile")
def _RiscWhileGrad(_, grad):
  return None, None
