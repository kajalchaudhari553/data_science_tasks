from pulp import LpMaximize, LpProblem, LpVariable, value

# Define the LP problem
model = LpProblem("Maximize_Profit", LpMaximize)

# Decision variables
tables = LpVariable("Tables", lowBound=0, cat='Integer')
chairs = LpVariable("Chairs", lowBound=0, cat='Integer')

# Objective function: Maximize profit
model += 50 * tables + 30 * chairs, "Total Profit"

# Constraints
model += 4 * tables + 3 * chairs <= 240, "Wood Constraint"
model += 2 * tables + 1 * chairs <= 100, "Labor Constraint"

model.solve()
print("Status:", model.status)
print("Optimal Solution:")
print(f"Tables to produce: {tables.value()}")
print(f"Chairs to produce: {chairs.value()}")
print(f"Maximum Profit: ${value(model.objective)}")

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 80, 400)
y1 = (240 - 4*x)/3
y2 = 100 - 2*x

plt.figure(figsize=(8,6))
plt.plot(x, y1, label='Wood Constraint')
plt.plot(x, y2, label='Labor Constraint')
plt.xlim((0, 80))
plt.ylim((0, 80))
plt.fill_between(x, np.minimum(y1, y2), alpha=0.3)
plt.scatter(tables.value(), chairs.value(), color='red', label='Optimal Point')
plt.xlabel('Tables')
plt.ylabel('Chairs')
plt.title('Feasible Region & Optimal Solution')
plt.legend()
plt.grid()
plt.show()
