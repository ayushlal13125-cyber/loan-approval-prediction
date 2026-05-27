import pandas as pd
import numpy as np
df=pd.read_csv("train.csv")

print(df.head())
print(df.describe())
print(df.isnull().sum())

df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df=df.drop('Dependents',axis=1)
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median()).astype(int)
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median()).astype(int)
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].median()).astype(int)
df = df.drop('Loan_ID', axis=1)

print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=['Gender','Married','Self_Employed','Property_Area']
for col in cols:
    df[col]=le.fit_transform(df[col])
df['Loan_Status']=df['Loan_Status'].map({'Y':1,'N':0})
df['Education']=df['Education'].map({'Graduate':1,'Not Graduate':0})
x=df.drop('Loan_Status',axis=1)

print(df.head())

y=df['Loan_Status']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,pred))
data=[[1,1,1,0,25000,10000,80,360,0,2]]

def predict(model,data):
    p=model.predict(data)
    if p[0]==1:
        print("loan approved")
    else:
     print("loan rejected")

predict(model,data)
