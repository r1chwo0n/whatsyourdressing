import numpy as np
import skfuzzy as fuzz #library of python
from skfuzzy import control as ctrl

# Input Variables [Antecedent = condition]
#np.arange(start,stop,step) =  สร้าง array ที่มีค่าเริ่มต้นที่ start, สิ้นสุดที่ stop - step, และเพิ่มค่าทีละ step
weather = ctrl.Antecedent(np.arange(0, 41, 1), 'weather') 
activity = ctrl.Antecedent(np.arange(0, 11, 1), 'activity')
location = ctrl.Antecedent(np.arange(0, 11, 1), 'location')

# Output Variables [Consequent = result]
upper_body = ctrl.Consequent(np.arange(0, 11, 1), 'upper_body')
lower_body = ctrl.Consequent(np.arange(0, 11, 1), 'lower_body')
shoes = ctrl.Consequent(np.arange(0, 11, 1), 'shoes')

# Define fuzzy membership functions for inputs and outputs as before
weather['cold'] = fuzz.trimf(weather.universe, [0, 0, 15])
weather['warm'] = fuzz.trimf(weather.universe, [10, 20, 30])
weather['hot'] = fuzz.trimf(weather.universe, [25, 40, 40])

activity['casual'] = fuzz.trimf(activity.universe, [0, 0, 5])
activity['formal'] = fuzz.trimf(activity.universe, [3, 5, 7])
activity['sports'] = fuzz.trimf(activity.universe, [6, 10, 10])

location['indoor'] = fuzz.trimf(location.universe, [0, 0, 5])
location['outdoor'] = fuzz.trimf(location.universe, [5, 10, 10])

upper_body['t-shirt'] = fuzz.trimf(upper_body.universe, [0, 0, 5])
upper_body['sweater'] = fuzz.trimf(upper_body.universe, [4, 6, 8])
upper_body['jacket'] = fuzz.trimf(upper_body.universe, [7, 10, 10])

lower_body['shorts'] = fuzz.trimf(lower_body.universe, [0, 0, 5])
lower_body['jeans'] = fuzz.trimf(lower_body.universe, [4, 6, 8])
lower_body['trousers'] = fuzz.trimf(lower_body.universe, [7, 10, 10])

shoes['sneakers'] = fuzz.trimf(shoes.universe, [0, 0, 5])
shoes['formal'] = fuzz.trimf(shoes.universe, [4, 6, 8])
shoes['sandals'] = fuzz.trimf(shoes.universe, [7, 10, 10])

# Rules
rule1 = ctrl.Rule(weather['cold'] & activity['casual'] & location['outdoor'],
                  (upper_body['jacket'], lower_body['jeans'], shoes['sneakers']))

rule2 = ctrl.Rule(weather['hot'] & activity['sports'] & location['outdoor'],
                  (upper_body['t-shirt'], lower_body['shorts'], shoes['sneakers']))

# Control System
clothing_ctrl = ctrl.ControlSystem([rule1, rule2])
clothing_simulation = ctrl.ControlSystemSimulation(clothing_ctrl)

# Mapping function from string to numeric
def map_input_to_numeric(weather_str, activity_str, location_str):
    weather_dict = {'cold': 10, 'warm': 20, 'hot': 35}
    activity_dict = {'casual': 2, 'formal': 5, 'sports': 8}
    location_dict = {'indoor': 2, 'outdoor': 8}

    return weather_dict[weather_str.lower()], activity_dict[activity_str.lower()], location_dict[location_str.lower()]

# Mapping function from numeric to string
def map_output_to_string(upper_body_val, lower_body_val, shoes_val):
    upper_body_dict = {0: 'T-shirt', 5: 'Sweater', 10: 'Jacket'}
    lower_body_dict = {0: 'Shorts', 5: 'Jeans', 10: 'Trousers'}
    shoes_dict = {0: 'Sneakers', 5: 'Formal Shoes', 10: 'Sandals'}

    # Round values to nearest defined range for mapping
    upper_body_val = round(upper_body_val / 5) * 5
    lower_body_val = round(lower_body_val / 5) * 5
    shoes_val = round(shoes_val / 5) * 5

    return upper_body_dict[upper_body_val], lower_body_dict[lower_body_val], shoes_dict[shoes_val]

# Example input strings
input_weather = 'cold'
input_activity = 'casual'
input_location = 'outdoor'

# Convert input strings to numeric
numeric_weather, numeric_activity, numeric_location = map_input_to_numeric(input_weather, input_activity, input_location)

# Set inputs for the simulation
clothing_simulation.input['weather'] = numeric_weather
clothing_simulation.input['activity'] = numeric_activity
clothing_simulation.input['location'] = numeric_location

# Compute the result
clothing_simulation.compute()

# Get numeric outputs and convert them to strings
upper_body_str, lower_body_str, shoes_str = map_output_to_string(clothing_simulation.output['upper_body'],
                                                                 clothing_simulation.output['lower_body'],
                                                                 clothing_simulation.output['shoes'])

# Print the results
print(f"Upper Body: {upper_body_str}")
print(f"Lower Body: {lower_body_str}")
print(f"Shoes: {shoes_str}")
