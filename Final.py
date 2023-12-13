# Importing Libraries
import pandas as pd
import streamlit as st
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import math
import plotly.graph_objects as go

#Seeding Random
random.seed(100)

s = pd.read_csv('social_media_usage.csv')

#Clean Function
def clean_sm(x):
    return np.where(x==1,x,x*0)

#Creating Blank DataFrame
ss = pd.DataFrame(index = range(s.shape[0]), columns=['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age'])

#Importing data to ss dataframe
ss.sm_li = clean_sm(s.web1h)
ss.income = s.income
ss.education = s.educ2
ss.parent = clean_sm(s.par)
ss.married = clean_sm(s.marital)
ss.female = clean_sm(s.gender-1)
ss.age = s.age

#dropping Missing Values
ss[ss.income > 9] = np.NaN
ss[ss.education > 8] = np.NaN
ss[ss.age > 98] = np.NaN
ss = ss.dropna()

#Creating target vector(y) and feature set (X)
y = ss.sm_li
X = ss.drop('sm_li', axis=1)

#Splitting Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)

#Instantiate logistic Regression
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)

print(ss.head)


## StreamLit collecting Variables
with st.sidebar:
    # Income
    income_st = st.selectbox('Income',
                        options = ['Less than $10,000',
                                    '$10,000 to under $20,000',
                                    '$20,000 to under $30,000',
                                    '$30,000 to under $40,000',
                                    '$40,000 to under $50,000',
                                    '$50,000 to under $75,000',
                                    '$75,000 to under $100,000',
                                    '$100,000 to under $150,000',
                                    '$150,000 or more?'])
    # Education
    education_st = st.selectbox("Education level",
                        options = ['Less than high school (Grades 1-8 or no formal schooling)',
                                    'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)',
                                    'High school graduate (Grade 12 with diploma or GED certificate)',
                                    'Some college, no degree (includes some community college)',
                                    'Two-year associate degree from a college or university',
                                    'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)',
                                    'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                                    'Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])
    #parent
    parent_st = st.radio(
    "Kids?",
    ('Yes','No'))
    #married
    married_st = st.radio(
    "Married?",
    ('Yes','No'))
    #female
    female_st = st.radio(
    "Gender?",
    ('Female','Male'))
    #age
    age = st.slider(label = 'Age',
                    min_value = 0,
                    max_value = 97,
                    value = 1)

## Converting Variable
#income
if income_st == 'Less than $10,000':
    income = 1
if income_st == '$10,000 to under $20,000':
    income_st = 2
if income_st == '$20,000 to under $30,000':
    income = 3
if income_st == '$30,000 to under $40,000':
    income = 4
if income_st == '$40,000 to under $50,000':
    income = 5
if income_st == '$50,000 to under $75,000':
    income = 6
if income_st == '$75,000 to under $100,000':
    income = 7
if income_st == '$100,000 to under $150,000':
    income = 8
if income_st == '$150,000 or more?':
    income = 9

#Education
if education_st == 'Less than high school (Grades 1-8 or no formal schooling)':
    education = 1
if education_st == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)':
    education = 2
if education_st == 'High school graduate (Grade 12 with diploma or GED certificate)':
    education = 3
if education_st == 'Some college, no degree (includes some community college)':
    education = 4
if education_st == 'Two-year associate degree from a college or university':
    education = 5
if education_st == 'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)':
    education = 6
if education_st == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
    education = 7
if education_st == 'Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)':
    education = 8
#parent
if parent_st == "No":
    parent = 0
if parent_st == "Yes":
    parent = 1
#married
if married_st == "No":
    married = 0
if married_st == "Yes":
    married = 1
#female
if female_st =='Female':
    female = 1
if female_st == 'Male':
    female = 0
#age

samp_data = {'income':[income],'education':[education],'parent':[parent],'married':[married],'female':[female],'age':[age]}
sample = pd.DataFrame(samp_data)



prediction = log_reg.predict(sample)
probability = log_reg.predict_proba(sample)

if prediction:
    prob = probability[0][1]
    gauge_title = "User is likely a LinkedIn User"
else:
    prob = probability[0][0]
    gauge_title = "User is likely NOT a LinkedIn User"



st.title('LinkedIn Logistic Model Prediction')

# Plot score on gauge plot
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob,
    title = {'text': f"{gauge_title}"},
    gauge = {"axis": {"range": [.5, 1]},
    "steps": [
    {"range": [.5, .66], "color":"red"},
    {"range": [.66, .83], "color":"gray"},
    {"range": [.83, 1], "color":"lightgreen"}
    ],
    "bar":{"color":"yellow"}}
    ))

st.plotly_chart(fig)

if prediction:
    st.write(f'Based on our model, we predict with {math.trunc(probability[0][1]*100)}% certainty, this person DOES uses LinkedIn.')
else: st.write(f'Based on our model, we predict with {math.trunc(probability[0][0]*100)}% certainty, this person DOES NOT use LinkedIn.')






