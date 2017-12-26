import numpy as nm

import pandas as pd

train=pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv',encoding='latin-1')

test=pd.read_csv('test1234.csv',encoding='latin-1')
del  test["Unnamed: 0"]
actualresult=pd.read_csv('result.csv',encoding='latin-1')
del  actualresult["Unnamed: 0"]

train.apply(lambda x: sum(x.isnull()),axis=0) 
train["Self_Employed"].fillna("No",inplace=True)
train["Gender"].fillna("Male",inplace=True)
train["Dependents"].fillna("0",inplace=True)
train["Property_Area"].fillna("Urban",inplace=True)
train["LoanAmount"].fillna(train["LoanAmount"].mean(),inplace=True)
train["Married"].fillna("Yes",inplace=True)
train["Credit_History"].fillna(1,inplace=True)


from sklearn.preprocessing import LabelEncoder

#Data Manipulation

#Data Manipulation
test.apply(lambda x: sum(x.isnull()),axis=0) 

test["Self_Employed"].fillna("No",inplace=True)
test["Gender"].fillna("Male",inplace=True)
test["Dependents"].fillna("0",inplace=True)
test["Property_Area"].fillna("Urban",inplace=True)
test["LoanAmount"].fillna(train["LoanAmount"].mean(),inplace=True)
test["Married"].fillna("Yes",inplace=True)
test["Credit_History"].fillna(1,inplace=True)


train["Total Income"]=train["ApplicantIncome"]+train["CoapplicantIncome"]
test["Total Income"]=train["ApplicantIncome"]+train["CoapplicantIncome"]

from sklearn.preprocessing import LabelEncoder
var_mod_train = [ 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'LoanAmount','Total Income', 'LoanAmount', 'Credit_History', 'Property_Area', 'Loan_Status']
var_mod_test = [ 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'LoanAmount','Total Income', 'LoanAmount', 'Credit_History', 'Property_Area']

le = LabelEncoder()
for i in var_mod_train:
    train[i] = le.fit_transform(train[i])
for i in var_mod_test:
    test[i] = le.fit_transform(test[i])
actualresult["Label_Result"]=le.fit_transform(actualresult["Label_Result"])




from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

model = RandomForestClassifier(n_estimators=100)
predictor_var = [ 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'LoanAmount','Total Income', 'LoanAmount', 'Credit_History', 'Property_Area']
outcome_var="Loan_Status"
model.fit(train[predictor_var],train[outcome_var])
predictions= model.predict(test[predictor_var])
c={1:"Y",0:"N"}
accuracy = metrics.accuracy_score(predictions,actualresult["Label_Result"])
#Accuracy Score
print("Accuracy")
print(accuracy)
result=pd.DataFrame({"Loadn_ID":test["Loan_ID"],"Status":predictions})
result["Status"]=result["Status"].map(c)
print(result)
