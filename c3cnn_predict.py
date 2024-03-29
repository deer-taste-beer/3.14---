import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
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

# CustomDataset과 모델 클래스 정의는 이전 코드에서 사용한 것과 동일하게 유지합니다.
# 예를 들어, 여기에 CustomDataset 클래스와 C3CNNModel 클래스 정의를 포함시킵니다.

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
csv_file = "C:\\Users\\poip8\\Desktop\\k\\GT2.csv"

# 데이터셋 로딩

dataset = CustomDataset(csv_file)
data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
# 모델 로딩
model_path = "C:\\Users\\poip8\\Desktop\\k\\trained_c3cnn_model.pth"
model = C3CNNModel(input_size=len(CustomDataset(csv_file).used_columns))  # input_size 조정 필요
model.load_state_dict(torch.load(model_path))
model.eval()
# 예측 및 실제값 준비
for inputs, labels in data_loader:
    with torch.no_grad():
        predictions = model(inputs).numpy().flatten()

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(labels.numpy(), label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()

# 그래프 저장
plt.savefig("C:\\Users\\poip8\\Desktop\\k\\model_predictions.png")
plt.show()
