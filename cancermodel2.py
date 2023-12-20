import pandas as pd 
import numpy as np 
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df  = pd.read_csv('cancer patient data sets.csv')

# Exploring the data
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.dtypes)

# dropping the patient id and the Index column
df = df.drop(['Patient Id', 'index'], axis=1)

# converting the level column from object to numerical
df['Level'] = [0 if level == 'Low' else 1 if level == 'Medium' else 3 for level in df['Level']]



# Split the data into features (x) and labels (y)
x = df.drop('Level', axis=1)
y = df['Level']
# print(x)
# print(y)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Creating a data pipeline
preprocessing_step = [('scaler',StandardScaler())]
model_steps = [('classifier',SVC())]
pipeline = Pipeline(preprocessing_step+model_steps)

# training the model
pipeline.fit(x_train,y_train)
# making predictions on the test set
yhat = pipeline.predict(x_test)
# print(yhat)
# print(y_test)


# Evaluating the model
results = classification_report(yhat,y_test)
print(results)

# testing with real data
test_data = [[33,1,2,4,5,4,3,2,2,4,3,2,2,4,3,4,2,2,3,1,2,3,4]]
real_result = pipeline.predict(test_data)
print(real_result[0])


# saving the model
model_filename = "lung_cancer2.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)