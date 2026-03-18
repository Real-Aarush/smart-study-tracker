import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

study_hours   = np.random.uniform(1, 10, n)
sleep_hours   = np.random.uniform(4, 9, n)
attendance    = np.random.uniform(50, 100, n)
prev_score    = np.random.uniform(40, 95, n)

# Score formula: weighted combination + noise
score = (
    3.5 * study_hours +
    2.0 * sleep_hours +
    0.3 * attendance +
    0.25 * prev_score +
    np.random.normal(0, 4, n)
).clip(30, 100)

df = pd.DataFrame({
    'study_hours': study_hours.round(1),
    'sleep_hours': sleep_hours.round(1),
    'attendance':  attendance.round(1),
    'prev_score':  prev_score.round(1),
    'final_score': score.round(1)
})

df.to_csv('data/student_data.csv', index=False)
print("✅ Dataset created:", df.shape)
print(df.head())