import pickle
import os
import joblib
# 读取 pkl 文件中的数据
data = joblib.load('output\\try_001--dui\\vibe_output.pkl')
lst = pickle.loads(data)
print(data)