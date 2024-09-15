import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy input variables
weather = ctrl.Antecedent(np.arange(0, 41, 1), 'weather')
activity = ctrl.Antecedent(np.arange(0, 6, 1), 'activity')
location = ctrl.Antecedent(np.arange(0, 2, 1), 'location')
gender = ctrl.Antecedent(np.arange(0, 2, 1), 'gender')

# Define fuzzy output variables
upper_body = ctrl.Consequent(np.arange(0, 9, 1), 'upper_body')
lower_body = ctrl.Consequent(np.arange(0, 8, 1), 'lower_body')

# Define fuzzy membership functions
weather['cold'] = fuzz.trimf(weather.universe, [0, 0, 20])
weather['warm'] = fuzz.trimf(weather.universe, [15, 25, 35])
weather['hot'] = fuzz.trimf(weather.universe, [30, 40, 40])

activity['work'] = fuzz.trimf(activity.universe, [0, 0, 1])
activity['exercise'] = fuzz.trimf(activity.universe, [1, 1, 2])
activity['leisure'] = fuzz.trimf(activity.universe, [2, 2, 3])
activity['travel'] = fuzz.trimf(activity.universe, [3, 3, 4])
activity['party'] = fuzz.trimf(activity.universe, [4, 4, 5])
activity['shopping'] = fuzz.trimf(activity.universe, [5, 5, 5])

location['indoor'] = fuzz.trimf(location.universe, [0, 0, 1])
location['outdoor'] = fuzz.trimf(location.universe, [1, 1, 1])

gender['male'] = fuzz.trimf(gender.universe, [0, 0, 1])
gender['female'] = fuzz.trimf(gender.universe, [1, 1, 1])

# Membership functions for upper_body
upper_body['t-shirt'] = fuzz.trimf(upper_body.universe, [0, 0, 1])
upper_body['shirt'] = fuzz.trimf(upper_body.universe, [1, 1, 2])
upper_body['polo'] = fuzz.trimf(upper_body.universe, [2, 2, 3])
upper_body['sweater'] = fuzz.trimf(upper_body.universe, [3, 3, 4])
upper_body['jacket'] = fuzz.trimf(upper_body.universe, [4, 4, 5])
upper_body['coat'] = fuzz.trimf(upper_body.universe, [5, 5, 6])
upper_body['blazer'] = fuzz.trimf(upper_body.universe, [6, 6, 7])
upper_body['dress'] = fuzz.trimf(upper_body.universe, [7, 8, 9])

# Membership functions for lower_body
lower_body['trousers'] = fuzz.trimf(lower_body.universe, [0, 0, 1])
lower_body['shorts'] = fuzz.trimf(lower_body.universe, [1, 1, 2])
lower_body['skirt'] = fuzz.trimf(lower_body.universe, [2, 2, 3])
lower_body['jeans'] = fuzz.trimf(lower_body.universe, [3, 3, 4])
lower_body['leggings'] = fuzz.trimf(lower_body.universe, [4, 4, 5])
lower_body['pants'] = fuzz.trimf(lower_body.universe, [5, 5, 6])
lower_body['none'] = fuzz.trimf(lower_body.universe, [6, 7, 8])

# 3. สร้างกฎฟัซซี่ (Fuzzy Rules)
rules = [
    ctrl.Rule(weather['cold'] & activity['work'] & (location['indoor'] | location['outdoor']) & gender['male'], (upper_body['coat'], lower_body['trousers'])),
    ctrl.Rule(weather['cold'] & activity['work'] & (location['indoor'] | location['outdoor'])  & gender['female'], (upper_body['coat'], lower_body['skirt'])),
    
    ctrl.Rule(weather['cold'] & activity['leisure'] & location['indoor'] & (gender['male'] | gender['female']), (upper_body['sweater'], lower_body['pants'])),
    ctrl.Rule(weather['cold'] & activity['leisure'] & location['outdoor'] & (gender['male'] | gender['female']), (upper_body['coat'], lower_body['pants'])),

    ctrl.Rule(weather['cold'] & activity['travel'] & (location['indoor'] | location['outdoor']) & (gender['male'] | gender['female']), (upper_body['jacket'], lower_body['jeans'])),
    
    ctrl.Rule(weather['cold'] & (activity['party'] | activity['shopping']) & (location['indoor'] | location['outdoor']) & (gender['male'] | gender['female']), (upper_body['coat'], lower_body['jeans'])),
    
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['work'] & location['outdoor'] & gender['male'] , (upper_body['blazer'], lower_body['trousers'])),
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['work'] & location['outdoor'] & gender['female'] , (upper_body['blazer'], lower_body['skirt'])),
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['work'] & location['indoor'] & gender['male'] , (upper_body['shirt'], lower_body['trousers'])),
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['work'] & location['indoor'] & gender['female'] , (upper_body['shirt'], lower_body['skirt'])),

    ctrl.Rule(weather['warm'] & activity['leisure'] & location['outdoor'] & gender['male'] , (upper_body['polo'], lower_body['pants'])),
    ctrl.Rule(weather['warm'] & activity['leisure'] & location['outdoor'] & gender['female'] , (upper_body['polo'], lower_body['skirt'])),
    ctrl.Rule(weather['hot'] & activity['leisure'] & location['outdoor'] & gender['male'] , (upper_body['t-shirt'], lower_body['jeans'])),
    ctrl.Rule(weather['hot'] & activity['leisure'] & location['outdoor'] & gender['female'] , (upper_body['t-shirt'], lower_body['skirt'])),
    
    ctrl.Rule((weather['hot'] | weather['warm']) & activity['leisure'] & location['indoor'] & (gender['female'] | gender['male']) , (upper_body['t-shirt'], lower_body['shorts'])),
    
    ctrl.Rule(weather['warm'] & activity['travel'] & (location['indoor'] | location['outdoor']) & gender['male'], (upper_body['shirt'], lower_body['jeans'])),
    ctrl.Rule(weather['warm'] & activity['travel'] & (location['indoor'] | location['outdoor']) & gender['female'], (upper_body['shirt'], lower_body['skirt'])),

    ctrl.Rule(weather['hot'] & activity['travel'] & (location['indoor'] | location['outdoor']) & gender['female'], (upper_body['sweater'], lower_body['jeans'])),
    ctrl.Rule(weather['hot'] & activity['travel'] & (location['indoor'] | location['outdoor']) & gender['male'], (upper_body['t-shirt'], lower_body['shorts'])),
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['party'] & (location['indoor'] | location['outdoor']) & gender['female'], (upper_body['dress'], lower_body['none'])),
    
    ctrl.Rule(weather['warm'] & activity['shopping'] & (location['indoor'] | location['outdoor']) & gender['male'] , (upper_body['polo'], lower_body['trousers'])),
    ctrl.Rule(weather['warm'] & activity['party'] & (location['indoor'] | location['outdoor']) & gender['male'] , (upper_body['shirt'], lower_body['trousers'])),
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['shopping'] & location['outdoor'] & gender['female'] , (upper_body['sweater'] , lower_body['skirt'])),
    ctrl.Rule((weather['warm'] | weather['hot']) & activity['shopping'] & location['indoor'] & gender['female'] , (upper_body['shirt'] , lower_body['shorts'])),
    ctrl.Rule(weather['hot'] & activity['party'] & (location['outdoor'] | location['indoor']) & gender['male'] , (upper_body['t-shirt'], lower_body['jeans'])),

    ctrl.Rule((weather['cold'] | weather['warm'] | weather['hot']) & activity['exercise'] & (location['indoor'] | location['outdoor']) & gender['female'] , (upper_body['t-shirt'], lower_body['leggings'])),
    ctrl.Rule((weather['cold'] | weather['warm'] | weather['hot'])& activity['exercise'] & (location['indoor'] | location['outdoor']) & gender['male'] , (upper_body['t-shirt'], lower_body['shorts'])),
]

# 4. สร้างระบบควบคุมฟัซซี่ (Control System)
clothing_ctrl = ctrl.ControlSystem(rules)
clothing_sim = ctrl.ControlSystemSimulation(clothing_ctrl)

def map_input_values(weather_input, activity_input, location_input, gender_input):
    # Mapping string inputs to numbers for fuzzy logic
    weather_value = float(weather_input)  # Weather as number (temperature)
    activity_map = {'work': 0, 'exercise': 1, 'leisure': 2, 'travel': 3, 'party': 4, 'shopping': 5}
    location_map = {'indoor': 0, 'outdoor': 1}
    gender_map = {'male': 0, 'female': 1}

    activity_value = activity_map.get(activity_input, 0)
    location_value = location_map.get(location_input, 0)
    gender_value = gender_map.get(gender_input, 0)

    return weather_value, activity_value, location_value, gender_value


def get_upper_body_recommendation(value):
    if 0 <= value < 1:
        return 'T-shirt'
    elif 1 <= value < 2:
        return 'Shirt'
    elif 2 <= value < 3:
        return 'Polo'
    elif 3 <= value < 4:
        return 'Sweater'
    elif 4 <= value < 5:
        return 'Jacket'
    elif 5 <= value < 6:
        return 'Coat'
    elif 6 <= value < 7:
        return 'Blazer'
    elif 7 <= value < 8:
        return 'Dress'
    else:
        return 'Unknown'

def get_lower_body_recommendation(value):
    if 0 <= value < 1:
        return 'Trousers'
    elif 1 <= value < 2:
        return 'Shorts'
    elif 2 <= value < 3:
        return 'Skirt'
    elif 3 <= value < 4:
        return 'Jeans'
    elif 4 <= value < 5:
        return 'Leggings'
    elif 5 <= value < 6:
        return 'Pants'
    elif 6 <= value < 7:
        return 'None'  # สำหรับกรณีที่ใส่ dress และไม่มีเสื้อผ้าส่วนล่าง
    else:
        return 'Unknown'

# Get inputs from terminal
weather_input = input("Enter weather (0-40): ")
activity_input = input("Enter activity (work, exercise, leisure, travel, party, shopping): ").lower()
location_input = input("Enter location (indoor or outdoor): ").lower()
gender_input = input("Enter gender (male or female): ").lower()

# Map the inputs to fuzzy values
weather_value, activity_value, location_value, gender_value = map_input_values(weather_input, activity_input, location_input, gender_input)

# Pass inputs to the fuzzy control system
clothing_sim.input['weather'] = weather_value
clothing_sim.input['activity'] = activity_value
clothing_sim.input['location'] = location_value
clothing_sim.input['gender'] = gender_value

# Compute the results
clothing_sim.compute()

# ตรวจสอบค่าที่ระบบฟัซซี่คำนวณได้
# print(f"Upper Body Raw Output: {clothing_sim.output['upper_body']}")
# print(f"Lower Body Raw Output: {clothing_sim.output['lower_body']}")

upper_clothing = get_upper_body_recommendation(clothing_sim.output['upper_body'])
lower_clothing = get_lower_body_recommendation(clothing_sim.output['lower_body'])
# Print the recommendation
print(f"Recommended upper_body: {upper_clothing}")
print(f"Recommended lower_body: {lower_clothing}")