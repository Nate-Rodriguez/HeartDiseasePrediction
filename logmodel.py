
import pandas as pd
import numpy as np
import streamlit as st
import sklearn


import csv
import os

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #Z-score variables
from sklearn.model_selection import train_test_split # simple TT split cv
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler #Z-score variables


#read in data 
data = pd.read_csv('Heart_Disease_Prediction.csv')
print(data)

#clean data
data = data.dropna()
data = pd.get_dummies(data, columns = ["Heart Disease"])


# Create logistic regression model
preds = ["Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120", "EKG results", "Max HR", "Exercise angina", 
         "ST depression", "Slope of ST", "Thallium", "Number of vessels fluro"]

cont = ["BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro"]

X = data[preds]
y = data["Heart Disease_Presence"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

# zscore
z = StandardScaler()
X_train[cont] = z.fit_transform(X_train[cont])
X_test[cont] = z.transform(X_test[cont])

# create and fit model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# ------------------------------------------

DATASET = data
LOGMODEL = lr

def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pd.DataFrame(np.sort(DATASET.values, axis=0),
                                index=DATASET.index,
                                columns=DATASET.columns)
        return heart_df


    def user_input_features() -> pd.DataFrame:
        age = st.sidebar.number_input("Age", min_value = 0, max_value = 100)
        sex = st.sidebar.number_input("Sex (1 = Male, 0 = Female)", 0, 1)
        chest_pain_type = st.sidebar.number_input("Respond with you Chest Pain Type (1 = typical angina, 2 =  atypical angina, 3 = non-anginal pain, 4 = asympotomatic)",
                                       1, 2, 3, 4)
        blood_pressure = st.sidebar.number_input("Enter Your Blood Pressure: ",
                                       min_value = 0, max_value = 300)
        chol = st.sidebar.number_input("Enter Your Cholesterol (mg/dl): ",
                                       min_value = 0, max_value = 300)
        fbs = st.sidebar.number_input("Is your fasting blood pressure over 120? (1 = yes, 0 = no) ",
                                       0, 1)
        ekg = st.sidebar.number_input("Resting Electrocardiographic Results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left vantricular hypertrophy): ",
                                       0,1,2)
        maxhr = st.sidebar.number_input("What is your max heart rate?", min_value = 0, max_value = 250)
        exrang = st.sidebar.number_input("Do you have exercise enduced angina (1 = yes, 0 = no)?", 0,1)
        oldpeak = st.sidebar.number_input("ST depression induced by exercise relative to rest", min_value = 0, max_value = 7)
        slopest = st.sidebar.number_input("What isthe slope of your peak exercise ST segment? (1 = upsloping, 2 = flat, 3 = downsloping)", 1,2,3)
        thal = st.sidebar.number_input("Thallium (3 = normal, 6 = fixed defect, 7 = reversable defect)", 3,6,7)
        numfluro = st.sidebar.number_input("Number of Vessels Fluro: ", 0,1,2,3)

        features = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "Chest Pain Type": [chest_pain_type],
            "Blood Pressure": [blood_pressure],
            "Cholesterol": [chol],
            "FBS over 120": [fbs],
            "EKG results": [ekg],
            "Max Heart Rate": [maxhr],
            "Exercise Angina": [exrang],
            "ST Depression": [oldpeak],
            "Slope of ST": [slopest],
            "Thallium": [thal],
            "Number of Vessels Fluro": [numfluro]
        })

        return features


    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="chart_with_upwards_trend"
    )

    st.title("Heart Disease Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.image("doctor.png",
    #              caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
    #              width=150)
    #     submit = st.button("Predict")
    # with col2:
    #     st.markdown("""
    #     Did you know that machine learning models can help you
    #     predict heart disease pretty accurately? In this app, you can
    #     estimate your chance of heart disease in seconds!
        
    #     Here, a logistic regression model using an undersampling technique
    #     was constructed using survey data of over 300k US residents from the year 2020.
    #     This application is based on it because it has proven to be better than the random forest
    #     (it achieves an accuracy of about 83%, which is quite good).
        
    #     To predict your heart disease status, simply follow the steps bellow:
    #     1. Enter the parameters that best describe you;
    #     2. Press the "Predict" button and wait for the result.
            
    #     **Keep in mind that this results is not equivalent to a medical diagnosis!
    #     This model would never be adopted by health care facilities because of its less
    #     than perfect accuracy, so if you have any problems, consult a human doctor.**
        
    #     **Author: Nate Rodriguez ([GitHub](https://github.com/kamilpytlak/heart-condition-checker))**
        
    #     You can see the steps of building the model, evaluating it, and cleaning the data itself
    #     on my GitHub repo [here](https://github.com/kamilpytlak/data-analyses/tree/main/heart-disease-prediction). 
    #     """)

    heart = load_dataset()

    st.sidebar.title("Feature Selection")
    st.sidebar.image("https://cdn.pixabay.com/photo/2012/04/01/18/57/heart-24037_1280.png", width=100)

    input_df = user_input_features()
    DF = pd.concat([input_df, heart], axis=0)
    DF = DF.drop(columns=["Heart Disease"])

    cat_cols = ["Age", "Sex", "Chest pain type", "BP", "Cholesterol",
                "FBS over 120", "EKG results", "Max HR", "Exercise Angina", "ST Depression",
                "Slope of ST", "Thallium", "Number of vessels fluro"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(DF[cat_col], prefix=cat_col)
        DF = pd.concat([DF, dummy_col], axis=1)
        del DF[cat_col]

    DF = DF[:1]
    DF.fillna(0, inplace=True)

    log_model = LOGMODEL

    if submit:
        prediction = log_model.predict(DF)
        prediction_prob = log_model.predict_proba(DF)
        if prediction == 0:
            st.markdown(f"**The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")
            st.image("images/heart-okay.jpg",
                     caption="Your heart seems to be okay! - Dr. Logistic Regression")
        else:
            st.markdown(f"**The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
            st.image("images/heart-bad.jpg",
                     caption="I'm not satisfied with the condition of your heart! - Dr. Logistic Regression")


if __name__ == "__main__":
    main()





