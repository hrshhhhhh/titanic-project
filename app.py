import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival")

df = pd.read_csv("train.csv")

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df.drop('Survived', axis=1)
y = df['Survived']

model = RandomForestClassifier()
model.fit(X, y)

st.subheader("Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)

sex = 0 if sex == "male" else 1

if st.button("Predict"):
    result = model.predict([[pclass, sex, age, fare]])
    
    if result[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")