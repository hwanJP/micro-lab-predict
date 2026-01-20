import pickle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# A.brasiliensis 28일 모델 분석 - 어떤 성분이 감소에 효과적인지 확인
print("=== A.brasiliensis 0-28 Feature Order ===")
print()

with open('LR_A.brasiliensis_0-28.pkl', 'rb') as f:
    obj = pickle.load(f)

model = obj['model']
features = obj['X']

# 인덱스와 feature, 계수 출력
print("Index | Feature | Coefficient")
print("-" * 50)
for i, (feat, coef) in enumerate(zip(features, model.coef_)):
    print(f"{i:5} | {feat} | {coef:.4f}")

print()
print(f"Intercept: {model.intercept_:.4f}")
print()

# 새 초기값 (방부제/항균/피부장벽 낮춤)
current_values = [4.70, 0.50, 8.00, 2.00, 2.00, 0.10, 0.00, 0.00, 0.00, 3.00, 0.00, 4.00, 0.00, 3.00, 2.00, 62.00, 0.00, 10.00, 1.00, 0.00, 0.00, 0.00, 0.00]

print("=== Current Initial Values ===")
for i, (feat, val, coef) in enumerate(zip(features, current_values, model.coef_)):
    contribution = val * coef
    print(f"{i:2}: {feat:30} = {val:6.2f} * {coef:8.4f} = {contribution:10.4f}")

# 예측값 계산
import numpy as np
pred = np.dot(current_values, model.coef_) + model.intercept_
print()
print(f"Predicted diff (28-day reduction): {pred:.4f}")
print(f"Initial log: {current_values[0]}")
print(f"Final log value: {current_values[0] - pred:.4f}")
print(f"Final CFU: {10**(current_values[0] - pred):.0f}")
