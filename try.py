import streamlit as st
import joblib
import numpy as np

model = joblib.load('Xgboost_class_Final.pkl')

def main():
    st.title('UTS Model Deployment_2602068675_MutiaraIndriaZ')
    CreditScore = st.slider('CreditScore', min_value=0, max_value=1000, value=1)
    Age = st.slider('Age', min_value=18, max_value=100, value=1)
    Gender = st.radio('Gender', ['Female', 'Male'])

    Gender_Female = 1 if Gender == 'Female' else 0
    Gender_Male = 1 if Gender == 'Male' else 0
    EstimatedSalary = st.slider('Estimated Salary', min_value=11.0, max_value=200000.0, value=0.1)
    Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
    Geography_France = 1 if Geography == 'France' else 0
    Geography_Germany = 1 if Geography == 'Germany' else 0
    Geography_Spain = 1 if Geography == 'Spain' else 0
    Tenure = st.slider('Tenure', min_value=0, max_value=10, value=1)
    Balance = st.slider('Balance', min_value=0.0, max_value=280000.0, value=0.1)
    NumOfProducts = st.slider('Num Of Products', min_value=1, max_value=4, value=1)
    HasCrCard = st.radio('Has Credit Card', ['Yes', 'No'])
    HasCrCard = 0 if HasCrCard == 'No' else 1
    IsActiveMember = st.radio('Is Active Member', ['Yes', 'No'])
    IsActiveMember = 0 if IsActiveMember == 'No' else 1
    if st.button('Make Prediction'):
        features = [CreditScore, Age, Gender_Female, Gender_Male, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, Geography_France, Geography_Germany, Geography_Spain, EstimatedSalary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')
def map_prediction(prediction):
    if prediction == 0:
        return "No Churn"
    else:
        return "Churn"
def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    mapped_prediction = map_prediction(prediction[0])
    return mapped_prediction

if __name__ == '__main__':
    main()
