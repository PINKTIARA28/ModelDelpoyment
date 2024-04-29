import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pickle

class ChurnPredictionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def explore_data(self):
        print("Shape of data:", self.df.shape)
        print("Columns:", self.df.columns)
        print("Value counts of 'Surname':\n", self.df['Surname'].value_counts())
        print("Value counts of 'Unnamed: 0':\n", self.df['Unnamed: 0'].value_counts())
        print(self.df.info())
        print("Missing values:\n", self.df.isnull().sum())

    def visualize_data(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.df['CreditScore'])
        plt.title('Box Plot of CreditScore')
        plt.show()

    def preprocess_data(self):
      df2 = self.df.copy()
      df2.drop(columns=['Unnamed: 0', 'Surname', 'id', 'CustomerId'], inplace=True)
      self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        df2.drop('churn', axis=1), df2['churn'], test_size=0.2, random_state=42
      )
    # Impute missing values
      imputer = SimpleImputer(strategy='median')
      self.x_train['CreditScore'] = imputer.fit_transform(self.x_train[['CreditScore']])
      self.x_test['CreditScore'] = imputer.transform(self.x_test[['CreditScore']])
    # One-hot encode categorical variables
      self.x_train = pd.get_dummies(self.x_train, columns=['Geography', 'Gender'])
      self.x_test = pd.get_dummies(self.x_test, columns=['Geography', 'Gender'])


    def train_model(self):
        self.model = xgb.XGBClassifier()
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        print(classification_report(self.y_test, y_pred, target_names=['0', '1']))

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)


# Usage
if __name__ == "__main__":
    churn_model = ChurnPredictionModel("/content/data_B.csv")
    churn_model.load_data()
    churn_model.explore_data()
    churn_model.visualize_data()
    churn_model.preprocess_data()
    churn_model.train_model()
    churn_model.evaluate_model()
    churn_model.save_model('Xgboost_class.pkl')
