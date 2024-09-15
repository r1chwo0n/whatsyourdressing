import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. สร้างตัวแปรฟัซซี่ (Fuzzy Variables)
# ตัวแปรอินพุต (Input Variables)
weather = ctrl.Antecedent(np.arange(0, 41, 1), 'weather')  # อุณหภูมิ 0 - 40 องศาเซลเซียส
activity = ctrl.Antecedent(np.arange(0, 11, 1), 'activity')  # ระดับกิจกรรม 0-10
location = ctrl.Antecedent(np.arange(0, 11, 1), 'location')  # ความเป็นทางการของสถานที่ 0-10

# ตัวแปรเอาท์พุต (Output Variables)
upper_body = ctrl.Consequent(np.arange(0, 11, 1), 'upper_body')  # แนะนำเสื้อผ้าท่อนบน
lower_body = ctrl.Consequent(np.arange(0, 11, 1), 'lower_body')  # แนะนำเสื้อผ้าท่อนล่าง
shoes = ctrl.Consequent(np.arange(0, 11, 1), 'shoes')  # แนะนำรองเท้า

# 2. กำหนดฟังก์ชันสมาชิก (Membership Functions)
# ฟังก์ชันสมาชิกสำหรับ weather
weather['cold'] = fuzz.trimf(weather.universe, [0, 0, 15])
weather['warm'] = fuzz.trimf(weather.universe, [10, 20, 30])
weather['hot'] = fuzz.trimf(weather.universe, [25, 40, 40])

# ฟังก์ชันสมาชิกสำหรับ activity
activity['low'] = fuzz.trimf(activity.universe, [0, 0, 5])
activity['medium'] = fuzz.trimf(activity.universe, [3, 5, 7])
activity['high'] = fuzz.trimf(activity.universe, [5, 10, 10])

# ฟังก์ชันสมาชิกสำหรับ location
location['casual'] = fuzz.trimf(location.universe, [0, 0, 5])
location['semi-formal'] = fuzz.trimf(location.universe, [3, 5, 7])
location['formal'] = fuzz.trimf(location.universe, [5, 10, 10])

# ฟังก์ชันสมาชิกสำหรับ upper_body
upper_body['t-shirt'] = fuzz.trimf(upper_body.universe, [0, 0, 3])
upper_body['shirt'] = fuzz.trimf(upper_body.universe, [2, 5, 8])
upper_body['jacket'] = fuzz.trimf(upper_body.universe, [7, 10, 10])

# ฟังก์ชันสมาชิกสำหรับ lower_body
lower_body['shorts'] = fuzz.trimf(lower_body.universe, [0, 0, 3])
lower_body['jeans'] = fuzz.trimf(lower_body.universe, [2, 5, 8])
lower_body['trousers'] = fuzz.trimf(lower_body.universe, [7, 10, 10])

# ฟังก์ชันสมาชิกสำหรับ shoes
shoes['sandals'] = fuzz.trimf(shoes.universe, [0, 0, 3])
shoes['sneakers'] = fuzz.trimf(shoes.universe, [2, 5, 8])
shoes['formal_shoes'] = fuzz.trimf(shoes.universe, [7, 10, 10])

# 3. สร้างกฎฟัซซี่ (Fuzzy Rules)
rule1 = ctrl.Rule(weather['cold'] & location['casual'], (upper_body['jacket'], lower_body['jeans'], shoes['sneakers']))
rule2 = ctrl.Rule(weather['hot'] & activity['high'], (upper_body['t-shirt'], lower_body['shorts'], shoes['sandals']))
rule3 = ctrl.Rule(weather['warm'] & location['formal'], (upper_body['shirt'], lower_body['trousers'], shoes['formal_shoes']))
rule4 = ctrl.Rule(weather['warm'] & activity['medium'], (upper_body['shirt'], lower_body['jeans'], shoes['sneakers']))

# 4. สร้างระบบควบคุมฟัซซี่ (Control System)
dressing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
dressing_sim = ctrl.ControlSystemSimulation(dressing_ctrl)

# 5. ให้ค่าข้อมูลอินพุต (Input)
dressing_sim.input['weather'] = 10.5 # อุณหภูมิ 28 องศาเซลเซียส
dressing_sim.input['activity'] = 7  # กิจกรรมระดับปานกลาง
dressing_sim.input['location'] = 4  # สถานที่ไม่ทางการ

# 6. คำนวณผลลัพธ์ (Compute)
dressing_sim.compute()

# ฟังก์ชันแปลงค่าผลลัพธ์ให้เป็นชนิดของเสื้อผ้าจริง ๆ
def get_upper_body_recommendation(value):
    if value <= 3:
        return 'T-shirt'
    elif 3 < value <= 7:
        return 'Shirt'
    else:
        return 'Jacket'

def get_lower_body_recommendation(value):
    if value <= 3:
        return 'Shorts'
    elif 3 < value <= 7:
        return 'Jeans'
    else:
        return 'Trousers'

def get_shoes_recommendation(value):
    if value <= 3:
        return 'Sandals'
    elif 3 < value <= 7:
        return 'Sneakers'
    else:
        return 'Formal Shoes'

# 7. แสดงผลลัพธ์แบบแปลงเป็นชนิดเสื้อผ้า
upper_body_type = get_upper_body_recommendation(dressing_sim.output['upper_body'])
lower_body_type = get_lower_body_recommendation(dressing_sim.output['lower_body'])
shoes_type = get_shoes_recommendation(dressing_sim.output['shoes'])

# 7. แสดงผลลัพธ์
print(f"upper_body: {upper_body_type}")
print(f"lower_body: {lower_body_type}")
print(f"shoes: {shoes_type}")

# ตรวจสอบค่าที่ระบบฟัซซี่คำนวณได้
print(f"Upper Body Raw Output: {dressing_sim.output['upper_body']}")
print(f"Lower Body Raw Output: {dressing_sim.output['lower_body']}")


