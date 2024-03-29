import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
# # 사용자 지정 데이터셋 클래스
# class CustomDataset(Dataset):
#     def __init__(self, csv_file):
#         #df = pd.read_csv(csv_file, skiprows=18, nrows=431995)  # 첫 18행을 제외하고 로드
#         df= pd.read_csv(r"C:\Users\poip8\Desktop\k\GT2.csv", header=None, skiprows=18, encoding='utf-8', low_memory=False)
#         self.input_columns = ['B', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'O', 'P', 'Q', 'R', 'S', 'T']
#         # 사용할 입력 변수 (C, E, M, N 제외, 여기서는 예를 들어 B도 제외합니다.)
#         self.used_columns = [col for col in self.input_columns if col not in [df.columns[0], df.columns[2], df.columns[4], df.columns[12], df.columns[13]]]
#         self.X = df.loc[:, self.used_columns].values
#         self.y = df.loc[:, 'E'].values
#         scaler = StandardScaler()
#         self.X = scaler.fit_transform(self.X)
        
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor([self.y[idx]], dtype=torch.float)

def evaluate_model(input_data, model, scaler):
    # 입력 데이터 스케일링
    scaled_data = scaler.transform(input_data)
    # 모델 예측 수행
    with torch.no_grad():
        model.eval()
        predictions = model(torch.tensor(scaled_data, dtype=torch.float)).numpy()
    return predictions



def sobol_sensitivity_analysis(model, dataset, column_names):
    problem = {
        'num_vars': len(column_names),
        'names': column_names,
        'bounds': [[-1.0, 1.0]] * len(column_names)
    }
    
    # Saltelli's sampling method를 사용하여 샘플 생성
    param_values = saltelli.sample(problem, 4096, calc_second_order=True)
    
    # 모델 평가
    Y = np.array([evaluate_model(np.array([params]), model, dataset.scaler) for params in param_values])
    
    # Sobol 분석 수행
    Si = sobol.analyze(problem, Y.flatten(), print_to_console=False)
    
    # 결과 시각화
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1차 Sobol 지수
    ax[0].bar(column_names, Si['S1'], color='skyblue')
    ax[0].set_title('First Order Sobol Indices')
    ax[0].set_ylabel('Sensitivity Index')
    ax[0].set_xticklabels(column_names, rotation=45, ha='right')
    
    # 총 Sobol 지수
    ax[1].bar(column_names, Si['ST'], color='skyblue')
    ax[1].set_title('Total Sobol Indices')
    ax[1].set_ylabel('Sensitivity Index')
    ax[1].set_xticklabels(column_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('sobol_sensitivity_analysis.png')
    plt.show()
    return Si



# 민감도 분석 함수 정의
def sensitivity_analysis(model, dataset, column_names):
    model.eval()  # 모델을 평가 모드로 설정
    
    # 모든 입력 변수에 대해 민감도 저장
    sensitivities = []
    
    # StandardScaler의 평균과 표준편차를 사용하여 역변환을 수행할 수 있습니다.
    scaler_mean = dataset.scaler.mean_
    scaler_std = dataset.scaler.scale_
    
    # 각 입력 변수에 대해 민감도 계산
    for i, name in enumerate(column_names):
        # 원본 데이터셋 복사
        X_modified = np.array(dataset.X)
        
        # i번째 변수에 작은 변화를 추가
        epsilon = scaler_std[i] * 0.01  # 1% 변화
        X_modified[:, i] += epsilon
        
        # 모델의 예측값 계산
        with torch.no_grad():
            predictions_original = model(torch.tensor(dataset.X, dtype=torch.float)).numpy().flatten()
            predictions_modified = model(torch.tensor(X_modified, dtype=torch.float)).numpy().flatten()
        
        # 변화율 계산
        change_rate = np.mean(np.abs(predictions_modified - predictions_original) / epsilon)
        sensitivities.append(change_rate)
         # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(column_names, sensitivities, color='skyblue')
    plt.xlabel('Input Variables')
    plt.ylabel('Sensitivity')
    plt.xticks(rotation=45)
    plt.title('Sensitivity Analysis of Input Variables')
    plt.tight_layout()  # 그래프가 잘리지 않도록 조정
    
    # 그래프 저장
    plt.savefig('sensitivity_analysis.png')
    plt.show()
    return sensitivities

def save_results_to_excel(column_names, sensitivities, sobol_indices, filename="sensitivity_results.xlsx"):
    # 일반 민감도 분석 결과 저장
    df_sensitivities = pd.DataFrame({
        'Variable': column_names,
        'Sensitivity': sensitivities,
    })
    
    # Sobol 지수 분석 결과 저장
    df_sobol = pd.DataFrame({
        'Variable': column_names,
        'S1': sobol_indices['S1'],
        'ST': sobol_indices['ST'],
    })
    
    # 엑셀 파일로 결과 저장
    with pd.ExcelWriter(filename) as writer:
        df_sensitivities.to_excel(writer, sheet_name='Sensitivity Analysis', index=False)
        df_sobol.to_excel(writer, sheet_name='Sobol Indices', index=False)


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        column_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        df = pd.read_csv(csv_file, header=None, skiprows=18, names=column_names, encoding='utf-8', low_memory=False)
        
        self.used_columns = ['B', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'O', 'P', 'Q', 'R', 'S', 'T']
        
        self.X = df.loc[:, self.used_columns].values
        self.y = df.loc[:, 'E'].values.astype(float)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor([self.y[idx]], dtype=torch.float)

class C3CNNModel(nn.Module):
    def __init__(self, input_size):
        super(C3CNNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x



# 모델 정의
class C3CNNModel(nn.Module):
    def __init__(self, input_size):
        super(C3CNNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 데이터셋 및 DataLoader 초기화
csv_file = "C:\\Users\\poip8\\Desktop\\k\\GT2.csv"
dataset = CustomDataset(csv_file)
train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
validation_size = len(dataset) - train_size - test_size
train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, validation_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# 모델, 손실 함수 및 최적화 알고리즘 초기화
model = C3CNNModel(input_size=len(dataset.used_columns))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Learning Start")

# 학습
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

print('훈련 완료')

# 모델 저장
model_save_path = "C:\\Users\\poip8\\Desktop\\k\\trained_c3cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
print("Model saved to", model_save_path)

# Sobol 민감도 분석 실행
sobol_indices = sobol_sensitivity_analysis(model, dataset, dataset.used_columns)

# 일반 민감도 분석 실행 및 결과 계산
sensitivities = sensitivity_analysis(model, dataset, dataset.used_columns)

# 결과를 엑셀 파일로 저장
save_results_to_excel(dataset.used_columns, sensitivities, sobol_indices, "C:\\Users\\poip8\\Desktop\\k\\sensitivity_results.xlsx")

# 누락된 변수 확인 및 출력
missing_columns = [col for col in dataset.input_columns if col not in dataset.used_columns]
if missing_columns:
    print("입력되지 않은 변수들:", missing_columns)
else:
    print("모든 변수가 입력되었습니다.")
