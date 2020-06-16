



import streamlit as st
import pandas as pd

st.write("""
# Simple admission Prediction App
This app predicts the chance of admit!
""")

df = pd.read_csv("Admission_Prediction.csv")

st.sidebar.header('User Input Parameters')


def user_input_features():
    GRE_score = st.sidebar.slider('GRE SCORE', 290, 340, 311)
    TOEFL_score = st.sidebar.slider('TOEFL Score', 92, 120, 101)
    #TOEFL_score = st.sidebar.slider('TOEFL score', 92.0, 120.0, 101.0)
    University_Rating = st.sidebar.slider('University Rating', 1, 5,3)
    SOP = st.sidebar.slider('SOP', 1.0, 5.0, 3.5)
    LOR = st.sidebar.slider('LOR', 1.0, 5.0, 2.5)
    CGPA = st.sidebar.slider('CGPA', 6.0, 10.0, 7.81)
    Research = st.sidebar.slider("Research",0.0,1.0,1.0)
       
        
    df1 = {'GRE_Score' : GRE_score,
            'TOEFL_Score' :  TOEFL_score,
            'University_Rating' : University_Rating,
            'SOP' : SOP,
            'LOR' : LOR,
            'CGPA' : CGPA,
            'Research' : Research}
    features = pd.DataFrame(df1, index=[0])
    return features

df1 = user_input_features()

st.subheader('User Input parameters')
st.write(df1)

# # dealing with missing values



df['TOEFL Score'].fillna(df['TOEFL Score'].mode()[0],inplace=True)
df['GRE Score'].fillna(df['GRE Score'].mode()[0],inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)


# dropping the 'Chance of Admit' and 'serial number' as they are not going to be used as features for prediction
x=df.drop(['Chance of Admit','Serial No.'],axis=1) 
# 'Chance of Admit' is the target column which shows the probability of admission for a candidate 
y=df['Chance of Admit'] 


# splitting the data into training and testing sets 
from sklearn.model_selection import train_test_split 
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2, random_state=100)


from sklearn import linear_model 
reg = linear_model.LinearRegression() 
reg.fit(train_x,train_y) 


prediction = reg.predict(df1)


#prediction = reg.predict(test_x)
#prediction_proba = reg.predict_proba(y)

from sklearn.metrics import r2_score 
score= r2_score(reg.predict(test_x),test_y)
print(score)




st.subheader('Chance of Admit')
st.write(prediction)










