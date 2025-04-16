# Python
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
