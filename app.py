import streamlit as st
import numpy as np
import joblib

model = joblib.load(
"svr_model.pkl"
)

scaler = joblib.load(
"scaler.pkl"
)

st.title(
"House Price Prediction SVR"
)

area = st.number_input(
"Area"
)

bedrooms = st.slider(
"Bedrooms",
1,10
)

bathrooms = st.slider(
"Bathrooms",
1,10
)

stories = st.slider(
"Stories",
1,5
)

parking = st.slider(
"Parking",
0,5
)

mainroad = st.selectbox(
"Main Road",
[0,1]
)

guestroom = st.selectbox(
"Guest Room",
[0,1]
)

basement = st.selectbox(
"Basement",
[0,1]
)

hotwater = st.selectbox(
"Hot Water Heating",
[0,1]
)

air = st.selectbox(
"Air Conditioning",
[0,1]
)

pref = st.selectbox(
"Preferred Area",
[0,1]
)

furnish = st.selectbox(
"Furnishing",
[0,1,2]
)

if st.button(
"Predict Price"
):

    data = np.array([[
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwater,
    air,
    parking,
    pref,
    furnish
    ]])

    data = scaler.transform(
    data
    )

    pred = model.predict(
    data
    )

    st.success(
    f"Predicted Price: ₹{pred[0]:,.0f}"
    )