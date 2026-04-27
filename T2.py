import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("train.csv")

# ---------------------------
# DATA CLEANING
# ---------------------------

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column
df.drop('Cabin', axis=1, inplace=True)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# ---------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------

plt.figure(figsize=(12, 8))

# 1. Survival Count
plt.subplot(2,3,1)
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")

# 2. Survival by Gender
plt.subplot(2,3,2)
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender")

# 3. Survival by Class
plt.subplot(2,3,3)
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival by Class")

# 4. Age Distribution
plt.subplot(2,3,4)
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")

# 5. Fare vs Survival
plt.subplot(2,3,5)
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")

plt.tight_layout()
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
