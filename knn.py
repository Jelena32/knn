import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

#Function to find the best k for K-Nearest Neighbors model based on the lowest MSE
def best_k(x_train,y_train):
      mse_values = []
      for k in range(1,21):
          knn = KNeighborsRegressor(n_neighbors=k)
          knn.fit(x_train, y_train)
          y_pred1 = knn.predict(x_train)
          mse = mean_squared_error(y_train, y_pred1)
          mse_values.append(mse)

      #Visualition of k and MSE
      plt.plot(range(1,21), mse_values)
      plt.show()
      plt.xlabel('k')
      plt.ylabel('MSE')
      return (np.argmin(mse_values)+1)

df = pd.read_excel('RSSI.xlsx', engine='openpyxl') # Read the Excel file 'RSSI.xlsx' into a pandas DataFrame
df.replace('*', -90, inplace=True) # Replace all occurrences of '*' in the DataFrame with the value -90,'*' represents missing value

#Split data in features and targets
x = df.iloc[:, :8].values
y = df.iloc[:, 8:10].values
y_scaled=y*0.1 #Scale targets values to represent them it in meters

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y_scaled, test_size=0.2, random_state=42)

k_best=best_k(x_train, y_train)

#Training model with best value of k
knn=KNeighborsRegressor(n_neighbors=k_best)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

#MSE value of test and predicted set
mse = mean_squared_error(y_test, y_pred)

#Visualization of actual test data (in blue) and the predicted data (in red)
y_testx=y_test[:,0]#X coordinate from test data
y_testy=y_test[:,1]#Y coordinate from test data
y_predictx=y_pred[:,0]#X coordinate from predicted data
y_predicty=y_pred[:,1]#Y coordinate from predicted values
plt.scatter(y_testx,y_testy,color="blue",label="Test values")
plt.scatter(y_predictx,y_predicty,color="red",label="Predicted values")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.show()
print(mse,k_best)










