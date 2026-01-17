from z3 import *

square = Int('square')
circle = Int('circle')
triangle = Int('triangle')

solver = Solver()

solver.add(square * square + circle == 16)
solver.add(triangle * triangle * triangle == 27)
solver.add(triangle * square == 6)

if solver.check() == sat:
    model = solver.model()
    print(model)