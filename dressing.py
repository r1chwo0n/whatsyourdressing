import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Define Fuzzy Variables
weather = ctrl.Antecedent(np.arange(0, 41, 1), 'weather')
occasion = ctrl.Antecedent(np.arange(0, 11, 1), 'occasion')
clothing = ctrl.Consequent(np.arange(0, 101, 1), 'clothing')

# Step 2: Define Membership Functions
weather['cold'] = fuzz.trimf(weather.universe, [0, 0, 15])
weather['mild'] = fuzz.trimf(weather.universe, [10, 20, 30])
weather['hot'] = fuzz.trimf(weather.universe, [25, 40, 40])

occasion['casual'] = fuzz.trimf(occasion.universe, [0, 0, 5])
occasion['semi-formal'] = fuzz.trimf(occasion.universe, [3, 5, 7])
occasion['formal'] = fuzz.trimf(occasion.universe, [5, 10, 10])

clothing['light'] = fuzz.trimf(clothing.universe, [0, 0, 40])
clothing['normal'] = fuzz.trimf(clothing.universe, [30, 50, 70])
clothing['heavy'] = fuzz.trimf(clothing.universe, [60, 100, 100])

# Step 3: Define Fuzzy Rules
rule1 = ctrl.Rule(weather['cold'] & occasion['formal'], clothing['heavy'])
rule2 = ctrl.Rule(weather['mild'] & occasion['casual'], clothing['normal'])
rule3 = ctrl.Rule(weather['hot'] & occasion['casual'], clothing['light'])
rule4 = ctrl.Rule(weather['hot'] & occasion['formal'], clothing['normal'])
rule5 = ctrl.Rule(weather['mild'] & occasion['formal'], clothing['normal'])
rule6 = ctrl.Rule(weather['cold'] & occasion['casual'], clothing['heavy'])

# Step 4: Create Control System
clothing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
clothing_simulation = ctrl.ControlSystemSimulation(clothing_ctrl)

# Step 5: Fuzzification & Simulation
clothing_simulation.input['weather'] = 18
clothing_simulation.input['occasion'] = 7

# Step 6: Compute & Defuzzification
clothing_simulation.compute()

# Output the result
print(f"Recommended clothing level: {clothing_simulation.output['clothing']:.2f}")
