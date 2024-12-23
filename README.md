Energy Consumption Prediction Project


This project aims to predict future energy consumption based on historical energy data using various machine learning techniques, including Linear Regression, Random Forest Regressor, and Neural Networks. The goal is to build and evaluate models that predict energy consumption (in MW) for a power utility company.

Dataset


The dataset used for this project is PJME_hourly.csv, which contains hourly energy consumption data for a specific power utility company. The key column of interest is PJME_MW, which represents the energy consumption in megawatts.

The dataset includes the following columns:

Datetime: Timestamp of the energy consumption record.
PJME_MW: The target variable representing energy consumption in megawatts.

Steps in the Project

1. Data Preprocessing

Convert the Datetime column to a datetime type and set it as the index.
Create additional time-based features (e.g., hour, day of the week, month, etc.) to help the models capture seasonal patterns.
Split the data into a training set (before 2015) and a test set (after 2015).
Scale the features using StandardScaler to normalize them for model training.


2. Model 1: Linear Regression
   
A simple linear regression model is trained to predict energy consumption based on the newly engineered features.


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Linear Regression model

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE Score on Test set: {rmse:.2f}')
RMSE on Test Set: 5683.95
3. Model 2: Random Forest Regressor
A Random Forest Regressor model is trained to predict the energy consumption. This model can handle complex patterns and interactions in the data better than linear regression.


from sklearn.ensemble import RandomForestRegressor

# Random Forest model

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE Score on Test set: {rmse:.2f}')
RMSE on Test Set: 1381.90
Feature Importances: The Random Forest model also provides insights into the most important features for predicting energy consumption.

4. Model 3: Neural Network Regression


A Neural Network model is created using TensorFlow/Keras, with two hidden layers and dropout regularization to reduce overfitting. The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Neural Network model

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
RMSE on Test Set: 5504.23
The model shows a similar RMSE score compared to the Linear Regression model but could be improved by tuning the model architecture.


5. Comparison of Models


The performance of each model is evaluated using the Root Mean Squared Error (RMSE) metric, which quantifies the error between the predicted and actual energy consumption values.

Model	RMSE (Test Set)
Linear Regression	5683.95
Random Forest	1381.90
Neural Network	5504.23
Best Model: The Random Forest Regressor offers the best performance, achieving a significantly lower RMSE compared to the other models.
Neural Network: The neural network shows an RMSE similar to the linear regression, but with potential for improvement in tuning.
Linear Regression: The linear regression model has the highest RMSE, indicating lower accuracy compared to the other models.

6. Model Evaluation Plots


For each model, the actual vs predicted energy consumption values are plotted for visual inspection of prediction accuracy.


import matplotlib.pyplot as plt

# Plotting Actual vs Predicted

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Sample Index')
plt.ylabel('Energy Consumption (MW)')
plt.legend()
plt.show()
Neural Network Architecture Description
Input Layer: The input layer corresponds to the features extracted from the dataset, with a shape equal to the number of features.
Hidden Layers: The model has two hidden layers:
The first hidden layer contains 64 neurons with ReLU activation.
The second hidden layer contains 32 neurons, also using ReLU activation.
Dropout Layer: A dropout rate of 50% is applied to the first hidden layer to prevent overfitting.
Output Layer: The output layer consists of a single neuron to predict the continuous value of energy consumption.
Optimizer: The model uses the Adam optimizer, which is efficient for training neural networks.
Loss Function: The model uses Mean Squared Error (MSE) for regression tasks.
Technologies Used
Python: For data manipulation and model building.
pandas: For data processing.
NumPy: For numerical operations.
Matplotlib/Seaborn: For visualizations.
scikit-learn: For machine learning models and preprocessing.
TensorFlow/Keras: For neural network model building and training.
How to Run the Project
Clone the repository:

git clone https://github.com/yourusername/energy-consumption-prediction.git
cd energy-consumption-prediction
Install the required dependencies:

pip install -r requirements.txt
Run the Jupyter notebook or Python script to execute the project.

Conclusion

This project demonstrates how different machine learning models can be used to predict energy consumption. Among the models tested, the Random Forest Regressor yielded the best performance, with a significantly lower RMSE compared to Linear Regression and Neural Network models. Future work can focus on improving the Neural Network by tuning hyperparameters or exploring other advanced algorithms for even better accuracy.
