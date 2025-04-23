#stock price prediction assignment
import pandas as pd
df = pd.read_csv('zomato_stock_price.csv')  #reading the csv file
print(df.head())
print(df.describe())
print(df.info())
#checking missing value
print(df.isnull().sum())
df['Date']= pd.to_datetime(df['Date'])  #converting date column to datetime type
df.set_index('Date', inplace=True)  #setting Date as index
print(df.head())
#select feature and target
x = df[['Open', 'High', 'Low', 'Volume']]  #feature
y = df[['Close']]  #target variable
#3. eda 
import matplotlib.pyplot as plt
import seaborn as sns
#Visualize the historical stock prices
plt.figure(figsize=(12, 8)) 
plt.plot(df['Close'],color = 'blue')  #plotting close price over time
plt.title('Historical Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
#Identify trends, seasonality between features
plt.figure(figsize=(14, 8)) 
plt.plot(df['Open'],label = 'Open')  #plotting close price over time
plt.plot(df['High'],label = 'High')
plt.plot(df['Low'],label = 'Low')
plt.plot(df['Close'],label = 'Close')
plt.title('Stock Price Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
#correlations between features
corr_matrix = df.corr()
print(df.corr())
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot= True, cmap= 'coolwarm', fmt= '.2f')
plt.title('Correlation Matrix')
plt.show()
#Plot scaNer plots to visualize rela'onships between features and the target variable
features = ['Open', 'High', 'Low', 'Volume']
for feature in features:
  plt.figure(figsize=(10,8))
  sns.scatterplot(x=feature, data= df, y='Close')
  plt.title(f'Scatter Plot of {feature} vs Close Price')
  plt.xlabel(feature)
  plt.ylabel('Close Price')
  plt.show()
# 4 Feature Engineering
#Create additional features 
df['MA_7'] = df['Close'].rolling(window=7).mean()  #7-day and 14-day moving averages
df['MA_14'] = df['Close'].rolling(window=14).mean()
print(df['MA_7'])
#creating Daily Percentage Change
df['Pct_change'] = df['Close'].pct_change()
print(df['Pct_change'])
#Split the dataset into training and tes'ng sets
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle= False)
#5 Model Building
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) #fit the model on training dataset
y_pred = lr.predict(x_test)  ## Predict on the test set
#6 evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('Model evaluation Metrics:')
print('Mean Squared Error:', mse)
print('R2 Score:', r2)
print('Mean Absolute Value:', mae)
#Plot Actual vs Predicted Close Prices(making prediction)
plt.figure(figsize=(12, 8))
plt.plot(y_test.index, y_test, label= 'Actual Close Price', color= 'blue')
plt.plot(y_test.index, y_pred, label= 'Predicted Close Price', color= 'red')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10,5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Close Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()
#7 colclusion:
#The linear regression model gives a R-squared = 0.9943833887099675, indicating that the model can explain % of the variance in the closing price.
#The MAE and MSE show the average magnitude of error. Mean Squared Error: 1.2839648404793076, Mean Absolute Value: 0.8275316195923991
