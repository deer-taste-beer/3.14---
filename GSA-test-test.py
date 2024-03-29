import pandas as pd
import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
#from tensorflow.keras.models import load_model
from keras.models import load_model
import tensorflow

import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 모델 불러오기
model_path = "C:\\Users\\poip8\\Desktop\\k\\my_bestMLP_model.h5"
model = load_model(model_path)

# 데이터 불러오기
csv_path = "C:\\Users\\poip8\\Desktop\\k\\GT2.csv"
data = pd.read_csv(csv_path, skiprows=1)  # 첫 번째 행을 헤더로 가정

# 입력 변수와 출력 변수 정의
X_columns = [col for col in data.columns if col != 'E']  # 'E' 열을 제외한 모든 열을 입력 변수로 사용
y_column = 'E'

X = data[X_columns].astype(float)  # 입력 변수
y = data[y_column].astype(float)  # 출력 변수 'E'

# 다중공선성 검사
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# VIF가 10 초과하는 열 제거
columns_to_drop = vif_data[vif_data['VIF'] > 10]['feature'].values
Xt = X.drop(columns=columns_to_drop)

# 수정된 입력 변수 이름 가져오기
names = Xt.columns

# Sobol 민감도 분석 파라미터 정의
problem = {
    'num_vars': Xt.shape[1],
    'names': names.tolist(),
    'bounds': [[0, 1]] * Xt.shape[1]
}

# Sobol 샘플 생성
samples = sobol_sample.sample(problem, 1024)  # 샘플 수를 줄여서 분석 시간 단축

# 모델 평가 함수
def evaluate_model(x):
    x = np.array(x).reshape(1, -1)  # 입력 형태 조정
    prediction = model.predict(x)
    return prediction.ravel()

# 모델 평가
outputs = np.array([evaluate_model(sample) for sample in samples]).flatten()

# Sobol 민감도 분석 수행
Sobol_indices = sobol_analyze.analyze(problem, outputs, print_to_console=False)

# 결과 그래프로 저장
fig, ax = plt.subplots()
ax.barh(names, Sobol_indices['S1'])
ax.set_xlabel('First-order Sobol indices')
ax.set_title('Sobol Sensitivity Analysis')
plt.tight_layout()
plt.savefig("C:\\Users\\poip8\\Desktop\\k\\Sobol_Sensitivity_Analysis.png")
