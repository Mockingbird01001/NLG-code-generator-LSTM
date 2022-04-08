
"""Proof that transforming (A*C)+(B*C) <=> (A+B)*C is "safe" if C=2^k.
Specifically, for all floating-point values A, B, and C, if
 - C is equal to +/- 2^k for some (possibly negative) integer k, and
 - A, B, C, A*C, B*C, and A+B are not subnormal, zero, or inf,
then there exists a rounding mode rm in [RTZ, RNE] such that
 (A*C) + (B*C) == (A+B) * C  (computed with rounding mode rm).
Informally, this means that the equivalence holds for powers of 2 C, modulo
flushing to zero or inf, and modulo rounding of intermediate results.
Requires z3 python bindings; try `pip install z3-solver`.
"""
import z3
FLOAT_TY = z3.Float16
a = z3.FP("a", FLOAT_TY())
b = z3.FP("b", FLOAT_TY())
c = z3.FP("c", FLOAT_TY())
s = z3.Solver()
s.add(z3.Extract(FLOAT_TY().sbits() - 1, 0, z3.fpToIEEEBV(c)) == 0)
for rm in [z3.RTZ(), z3.RNE()]:
  z3.set_default_rounding_mode(rm)
  before = a * c + b * c
  after = (a + b) * c
  s.add(
      z3.Not(
          z3.Or(
              z3.And(z3.fpIsZero(before), z3.fpIsZero(after)))))
  for x in [
      (a * c),
      (b * c),
      (a + b),
  ]:
    s.add(z3.Not(z3.fpIsSubnormal(x)))
    s.add(z3.Not(z3.fpIsZero(x)))
    s.add(z3.Not(z3.fpIsInf(x)))
if s.check() == z3.sat:
  m = s.model()
  print("Counterexample found!")
  print(m)
  print("a*c:       ", z3.simplify(m[a] * m[c]))
  print("b*c:       ", z3.simplify(m[b] * m[c]))
  print("a+b:       ", z3.simplify(m[a] + m[b]))
  print("a*c + b*c: ", z3.simplify(m[a] * m[c] + m[b] * m[c]))
  print("(a+b) * c: ", z3.simplify((m[a] + m[b]) * m[c]))
else:
  print("Proved!")
