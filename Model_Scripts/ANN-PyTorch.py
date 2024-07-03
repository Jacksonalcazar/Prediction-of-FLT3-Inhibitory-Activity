import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Setting seeds for reproducibility
np.random.seed(11)
torch.manual_seed(11)

# 1. Load the data
data = pd.read_csv('Data.csv', delimiter=';')
data = data.dropna()

# 2. Define features and target variable
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 4. Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 5. Create DataLoader
train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)

# 6. Create the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train_t.shape[1], 500)
        self.fc2 = nn.Linear(500, 5)
        self.fc3 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork()

# 7. Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 8. Train the model
for epoch in range(100):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# 9. Evaluation
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t)
    y_test_pred = model(X_test_t)

    train_r2 = r2_score(y_train, y_train_pred.numpy())
    test_r2 = r2_score(y_test, y_test_pred.numpy())
    train_mae = mean_absolute_error(y_train, y_train_pred.numpy())
    test_mae = mean_absolute_error(y_test, y_test_pred.numpy())
    train_std = np.std(y_train - y_train_pred.numpy().flatten())
    test_std = np.std(y_test - y_test_pred.numpy().flatten())
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred.numpy()))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred.numpy()))

    # 10. Prepare the metrics text
    metrics_text = (
        f"R2 Training: {train_r2}\n"
        f"R2 Test: {test_r2}\n"
        f"MAE Training: {train_mae}\n"
        f"MAE Test: {test_mae}\n"
        f"Standard Deviation Training: {train_std}\n"
        f"Standard Deviation Test: {test_std}\n"
        f"RMSE Training: {train_rmse}\n"
        f"RMSE Test: {test_rmse}\n"
    )

    # 11. Write the metrics to a .txt file
    with open('metrics_ANN-P.txt', 'w') as file:
        file.write(metrics_text)

# 12. Create dataframes for training and testing sets
train_results = pd.DataFrame({
    'Y_Train_Actual': y_train,
    'Y_Train_Predicted': y_train_pred.numpy().flatten()
})
test_results = pd.DataFrame({
    'Y_Test_Actual': y_test,
    'Y_Test_Predicted': y_test_pred.numpy().flatten()
})

# 13. Combine the results
combined_results = pd.concat([train_results.reset_index(drop=True), test_results.reset_index(drop=True)], axis=1)

# 14. Export to CSV
combined_results.to_csv('XY_ANN-P.csv', sep=';', index=False)
