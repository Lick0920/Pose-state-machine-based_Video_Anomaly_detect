import pickle
import os
import joblib
# ��ȡ pkl �ļ��е�����
data = joblib.load('output\\try_001--dui\\vibe_output.pkl')
lst = pickle.loads(data)
print(data)