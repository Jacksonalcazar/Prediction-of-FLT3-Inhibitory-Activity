import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load the data from a .csv file
data = pd.read_csv('Data.csv', delimiter=';')

# 2. Define features and target variable
X = data.iloc[:, 3:-1]
y = data.iloc[:, -1]

# 3. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 4. Train the SVM model with the training set
svm_model = SVR()
svm_model.fit(X_train, y_train)

# 5. Evaluate the model's performance
# a. R^2 coefficient of determination
train_r2 = r2_score(y_train, svm_model.predict(X_train))
test_r2 = r2_score(y_test, svm_model.predict(X_test))

# b. Mean absolute error (MAE)
train_mae = mean_absolute_error(y_train, svm_model.predict(X_train))
test_mae = mean_absolute_error(y_test, svm_model.predict(X_test))

# c. Standard deviation
train_std = np.std(y_train - svm_model.predict(X_train))
test_std = np.std(y_test - svm_model.predict(X_test))

# d. Root mean squared error (RMSE)
train_rmse = np.sqrt(mean_squared_error(y_train, svm_model.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, svm_model.predict(X_test)))

# 6. Prepare the metrics text
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

# 7. Write the metrics to a .txt file
with open('metrics_SVM.txt', 'w') as file:
    file.write(metrics_text)

# 8. Generate predicted values
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

# 9. Create dataframes for the training and testing sets with actual and predicted values
train_results = pd.DataFrame({
    'Y_Train_Actual': y_train,
    'Y_Train_Predicted': y_train_pred
})

test_results = pd.DataFrame({
    'Y_Test_Actual': y_test,
    'Y_Test_Predicted': y_test_pred
})

# 10. Combine the training and testing results into a single dataframe
combined_results = pd.concat([train_results.reset_index(drop=True), test_results.reset_index(drop=True)], axis=1)

# 11. Export the combined dataframe
combined_results.to_csv('XY_SVM.csv', sep=';', index=False)
