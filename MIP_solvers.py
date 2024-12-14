from pyomo.environ import (
    ConcreteModel, Var, Objective, ConstraintList, SolverFactory, NonNegativeReals, Binary, maximize
)
from gekko import GEKKO
import numpy as np

import numpy as np
import math
from pyomo.environ import *


def ipopt_solver(V_H, E_H, N, I, hyperedges):
    # Step 2: Create Pyomo model
    model = ConcreteModel()

    # Step 3: Define binary decision variables for each link (b_i)
    model.b = Var(range(V_H), domain=Binary)  # b[i] = 1 if link i is active, 0 otherwise

    # Step 4: Define the objective function (maximize throughput)
    def throughput(model):
        total_throughput = 0
        for i in range(V_H):
            interference = sum(model.b[j] * I[i, j] for j in range(V_H) if j != i)  # Interference from other active links
            denominator = N[i] + interference  # Noise + interference
            total_throughput += log(1+((I[i,i] * model.b[i]) / denominator))/log(2)  # Contribution of link i to throughput
        return total_throughput

    model.obj = Objective(rule=throughput, sense=maximize)

    # Step 5: Add constraints for hyperedges (each hyperedge has a threshold)
    model.constraints = ConstraintList()

    for e_idx, hyperedge in enumerate(hyperedges):
        non_zero_idx = np.nonzero(hyperedge)[0]
        if len(non_zero_idx) == 1:
            model.constraints.add(
                model.b[non_zero_idx[0]] <= 1
            )
        else:
            model.constraints.add(
                sum(model.b[i] for i in non_zero_idx) <= len(non_zero_idx)-1
            )

    # Step 6: Solve the model
    solver = SolverFactory('apopt')  # Use Ipopt for nonlinear problems
    result = solver.solve(model, tee=False)

    # Step 7: Extract the results
    optimal_decisions = [model.b[i].value for i in range(V_H)]

    # Output the optimal link schedule
    print("Optimal link schedule:", optimal_decisions)
    print("Maximum throughput:", model.obj())

    return model.obj()



def gekko_apopt_solver(V_H, E_H, N, I, hyperedges):
    # Initialize GEKKO model
    model = GEKKO(remote=False)
    model.options.SOLVER = 1  # Use APOPT solver

    # Step 3: Define binary decision variables for each link (b_i)
    b = [model.Var(value=0, integer=True, lb=0, ub=1) for _ in range(V_H)]  # Binary variables

    # Step 4: Define the objective function (maximize throughput)
    total_throughput = 0
    for i in range(V_H):
        interference = sum(b[j] * I[i, j] for j in range(V_H) if j != i)  # Interference from other active links
        denominator = N[i] + interference  # Noise + interference
        total_throughput += model.log(1 + (I[i, i] * b[i]) / denominator) / model.log(2)  # Contribution of link i
    model.Obj(-total_throughput)  # GEKKO minimizes by default, so negate the objective

    # Step 5: Add constraints for hyperedges (each hyperedge has a threshold)
    for hyperedge in hyperedges:
        non_zero_idx = np.nonzero(hyperedge)[0]
        if len(non_zero_idx) == 1:
            model.Equation(b[non_zero_idx[0]] <= 1)
        else:
            model.Equation(sum(b[i] for i in non_zero_idx) <= len(non_zero_idx) - 1)

    # Step 6: Solve the model
    model.solve(disp=False)

    # Step 7: Extract the results
    optimal_decisions = [b[i].value[0] for i in range(V_H)]

    # Output the optimal link schedule
    #print("Optimal link schedule:", optimal_decisions)
    #print("Maximum throughput:", -model.options.OBJFCNVAL)  # Negate to get the original objective value

    return -model.options.OBJFCNVAL

