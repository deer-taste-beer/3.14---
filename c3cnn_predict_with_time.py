import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        column_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        df = pd.read_csv(csv_file, header=None, skiprows=18, names=column_names, encoding='utf-8', low_memory=False)
        
        self.used_columns = ['B', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'O', 'P', 'Q', 'R', 'S', 'T']
        self.X = df.loc[:, self.used_columns].values
        self.y = df.loc[:, 'E'].values.astype(float)
        self.a = df.loc[:, 'A'].values  # 시계열 데이터 저장
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor([self.y[idx]], dtype=torch.float), self.a[idx]

class C3CNNModel(nn.Module):
    def __init__(self, input_size):
        super(C3CNNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 데이터셋 및 DataLoader 설정
csv_file = "C:\\Users\\poip8\\Desktop\\k\\GT2.csv"
dataset = CustomDataset(csv_file)
data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# 모델 로딩
model_path = "C:\\Users\\poip8\\Desktop\\k\\trained_c3cnn_model.pth"
model = C3CNNModel(input_size=len(dataset.used_columns))
model.load_state_dict(torch.load(model_path))
model.eval()

# 예측 및 실제값 준비
a_values = []  # 시계열 데이터를 저장할 리스트
predictions_list = []  # 예측값을 저장할 리스트
labels_list = []  # 실제값을 저장할 리스트

for inputs, labels, a in data_loader:
    with torch.no_grad():
        predictions = model(inputs).numpy().flatten()
        predictions_list.extend(predictions)
        labels_list.extend(labels.numpy().flatten())
    a_values.extend(a)  # a는 이미 배치마다 numpy 배열로 변환 가능한 상태여야 합니다.

# numpy 배열로 변환
predictions_array = np.array(predictions_list)
labels_array = np.array(labels_list)
a_values_array = np.array(a_values)  # 시계열 데이터인 A 열

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(a_values_array, labels_array, label='Actual', color='blue')
plt.plot(a_values_array, predictions_array, label='Predicted', color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Time (Seconds)')
plt.ylabel('ppm')
plt.legend()
plt.tight_layout()

# 그래프 저장
plt.savefig("C:\\Users\\poip8\\Desktop\\k\\model_predictions_with_time.png")
plt.show()
