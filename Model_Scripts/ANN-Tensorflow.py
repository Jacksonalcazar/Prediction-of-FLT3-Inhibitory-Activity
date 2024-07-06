import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import HeNormal
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import random

np.random.seed(11)
random.seed(11)
tf.random.set_seed(11)

# 1. Load the data from a .csv file
data = pd.read_csv('Data.csv', delimiter=';')

# 2. Data cleaning: Remove or impute NaN values
data = data.dropna()

# 3. Define features and target variable
X = data.iloc[:, 2:-1]
y = data.iloc[:, -1]

# 4. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 5. Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Create the Neural Network model
initializer = HeNormal(seed=11)
model = Sequential()
model.add(Dense(500, input_dim=X_train_scaled.shape[1], activation='relu', kernel_initializer=initializer))
model.add(Dense(5, activation='relu', kernel_initializer=initializer))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# 7. Train the model with the training set
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10)

# 8. Evaluate the model's performance
# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# a. R^2 coefficient of determination
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# b. Mean absolute error (MAE)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# c. Standard deviation
train_std = np.std(y_train - y_train_pred.flatten())
test_std = np.std(y_test - y_test_pred.flatten())

# d. Root mean squared error (RMSE)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# 9. Prepare the metrics text
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

# 10. Write the metrics to a .txt file
with open('metrics_ANN-T.txt', 'w') as file:
    file.write(metrics_text)

# 11. Create dataframes for the training and test sets with actual and predicted values
train_results = pd.DataFrame({
    'Y_Train_Actual': y_train,
    'Y_Train_Predicted': y_train_pred.flatten()
})

test_results = pd.DataFrame({
    'Y_Test_Actual': y_test,
    'Y_Test_Predicted': y_test_pred.flatten()
})

# 12. Combine the training and test results into a single dataframe
combined_results = pd.concat([train_results.reset_index(drop=True), test_results.reset_index(drop=True)], axis=1)

# 13. Export the combined dataframe
combined_results.to_csv('XY_ANN-T.csv', sep=';', index=False)
