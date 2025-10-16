import os
import numpy as np
import pandas as pd

# Configuration
n_weeks = 52
min_rows_per_week = 80
max_rows_per_week = 150
fraud_ratio = 0.01
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

# Create folder if not exists
os.makedirs('dataset/weeks', exist_ok=True)

# Weeks where we inject drift
drift_weeks = np.random.choice(range(1, n_weeks+1), size=5, replace=False).tolist()

for week in range(1, n_weeks+1):
    rows_per_week = np.random.randint(min_rows_per_week, max_rows_per_week + 1)
    week_data = []

    for _ in range(rows_per_week):
        time = float(week)
        V = np.random.normal(0, 1, 28)
        amount = np.random.exponential(100)
        class_label = 1 if np.random.rand() < fraud_ratio else 0

        if week in drift_weeks:
            # Extreme drift
            V[:5] += np.random.uniform(8, 12)       
            V[5:10] = np.random.normal(10, 5, 5)   
            amount *= np.random.uniform(2, 4)      
            class_label = 1 if np.random.rand() < 0.15 else 0  

        row = [time] + V.tolist() + [amount, class_label]
        week_data.append(row)

    week_df = pd.DataFrame(week_data, columns=columns)
    week_df.to_csv(f'dataset/weeks/week_{week}.csv', index=False)

print("Weekly datasets with varying lengths generated under dataset/weeks")
print("Artificial drift injected in weeks:", drift_weeks)
