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

sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)

# Fare slider: minimum is 50
fare = st.slider("Fare (min: $50)", min_value=50, max_value=500, value=50, step=1)

# Auto-determine Pclass based on fare
# Fare >= 200 → 1st Class, 100–199 → 2nd Class, 50–99 → 3rd Class
if fare >= 200:
    pclass = 1
    class_label = "1st Class (Luxury)"
elif fare >= 100:
    pclass = 2
    class_label = "2nd Class (Comfort)"
else:
    pclass = 3
    class_label = "3rd Class (Economy)"

st.info(f"🎫 **Passenger Class auto-set to: {class_label}** based on fare ${fare}")

sex_encoded = 0 if sex == "male" else 1

if st.button("Predict"):
    result = model.predict([[pclass, sex_encoded, age, fare]])

    if result[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
