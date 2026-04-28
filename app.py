import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("ML CI/CD Demo App")

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

st.write("Adjust input values:")

sl = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
pw = st.slider("Petal Width", 0.1, 2.5, 1.0)

if st.button("Predict"):
    data = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")
