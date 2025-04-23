#eda, data wrangling, handling missing value, outlier detection, univariable, bivariate, multivariate analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#load dataset available in seaborn library
df = sns.load_dataset('titanic')
print(df.head())
#eda
print(df.info())
#descriptive statistics
print(df.describe())
#visualizing the distribution of age to understand how age is distributed among passanger
sns.histplot(df['age'].dropna(), kde= True)
plt.title('Age Distribution')
plt.show()
#checking howmany missing value
print(df.isnull().sum())
#filling missing value'age' with median of column
df['age'].fillna(df['age'].median(), inplace = True)
#dropping row where 'embarked' is missing
df.dropna(subset=['embarked'], inplace= True)
#dropping the 'cabin' column as it has too many missing value
df.drop(columns='deck', inplace=True)
#univariant analysis
#plot distribution of age
sns.histplot(df['age'], kde= True)
plt.title('Distribution of Age')
plt.show()
#plot distribution of fare
sns.histplot(df['fare'], kde=True)
plt.title('Distribution of Fare')
plt.show()
#scatter plot between fare and survived
sns.boxplot(x='survived', y='fare', data= df)
plt.title('Fare vs Survived')
plt.show()
#check the correlation between fare and survived
correlation = df['fare'].corr(df['survived'])
print(f'Correlation between Fare and Survived:{correlation}')
#multivariante analysis
#paiplot for multiple feature
sns.pairplot(df[['age', 'fare', 'survived', 'pclass']], hue = 'survived')
plt.show()
#correlation heatmap
sns.heatmap(df[['age', 'fare', 'survived', 'pclass']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#step 1 : select only numeric column
numerical_columns=['age', 'survived', 'parch','sibsp','fare']
df_numerical = df[numerical_columns]
#drop rows for missing values
df_numerical = df_numerical.dropna()
#define features (x) and target (y)
x = df_numerical.drop(columns=['survived'])
y = df_numerical['survived']
#step 2: split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state= 42)
#step 3: scale the numeric features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#step 4: initialize the logistic regression model
logreg = LogisticRegression(max_iter=500)
#step 5: apply RFE with logistic regression to select the top 2 feature
rfe = RFE(logreg, n_features_to_select=2)
rfe.fit(x_train_scaled, y_train)
#step 6: check which feature were selected
print('Selected Features (True means selected):', rfe.support_)
print('Feature Ranking(1 means selected):', rfe.ranking_)
#step 7: train logistic regression using only the selected feature
x_train_rfe = rfe.transform(x_train_scaled)
x_test_rfe = rfe.transform(x_test_scaled)
logreg.fit(x_train_rfe, y_train)
#step 8: make predictions and evaluate the model
y_pred = logreg.predict(x_test_rfe)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acuracy of logistic regression with rfe on numeric features: {accuracy}')