import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as analyze_sobol
from scipy.interpolate import interp1d

# 엑셀 파일 경로
excel_file = r'C:\Users\poip8\Desktop\drive-download-20240313T030339Z-001 (2)\GT2.xlsx'

# 데이터 불러오기
df = pd.read_excel(excel_file, header=None, usecols="B:T", skiprows=18)

# 이상치 제거
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 결측치 보간
df.interpolate(method='linear', inplace=True)

# 분석 대상 열 (E 열) 분리
target_col = df.iloc[:, -1]
df.drop(df.columns[-1], axis=1, inplace=True)

# 데이터 준비
problem = {
    'num_vars': df.shape[1],
    'names': [f'X{i}' for i in range(df.shape[1])],
    'bounds': [[df[col].min(), df[col].max()] for col in df.columns]
}

# 데이터 준비
param_values = sobol.sample(problem, 1000, calc_second_order=False)

# 결과 계산
Y = target_col.values
sobol_indices = analyze_sobol(problem, Y, calc_second_order=False)

# 결과 출력
print("First order indices:")
for name, value in zip(problem['names'], sobol_indices['S1']):
    print(f"{name}: {value}")

print("\nTotal order indices:")
for name, value in zip(problem['names'], sobol_indices['ST']):
    print(f"{name}: {value}")
