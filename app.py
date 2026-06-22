import pandas as pd 
import pickle as pk 
import streamlit as st

try:
    model = pk.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please check the file path.")
    st.stop()

try:
    cars_data = pd.read_csv('Cardetails.csv')
except FileNotFoundError:
    st.error("Dataset 'Cardetails.csv' not found. Please check the file path.")
    st.stop()

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

st.markdown(
    """
    <style>
    body {background-color: #000000; color: #FFFFFF;}
    .stApp {background-color: #000000;}
    .title {font-size:35px !important; font-weight: bold; color: #32CD32; text-align: center;}
    .sub-title {font-size:20px !important; color: #FFD700; text-align: center;}
    .predict-button {background-color: #32CD32 !important; color: white !important; font-size: 20px !important; border-radius: 10px;}
    .prediction-result {color: #FFD700; font-size: 25px; font-weight: bold; text-align: center;}
    label {color: #FFFFFF !important;}
    .stSlider>div>div>div { background: #32CD32 !important; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="title">Parvani Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Estimate your car’s price with AI</p>', unsafe_allow_html=True)
st.markdown("---")


name = st.selectbox('Select Car Brand', cars_data['name'].unique())

col1, col2 = st.columns(2)
with col1:
    year = st.slider('Car Manufactured Year', 1994, 2024, 2015)
    km_driven = st.slider('No of Kms Driven', 100, 200000, 30000)
    mileage = st.slider('Car Mileage (kmpl)', 10, 40, 20)
with col2:
    engine = st.slider('Engine Capacity (CC)', 700, 5000, 1500)
    max_power = st.slider('Max Power (bhp)', 0, 200, 100)
    seats = st.slider('No of Seats', 2, 10, 5)

fuel = st.radio('Fuel Type', cars_data['fuel'].unique())
seller_type = st.radio('Seller Type', cars_data['seller_type'].unique())
transmission = st.radio('Transmission Type', cars_data['transmission'].unique())
owner = st.radio('Owner Type', cars_data['owner'].unique())

# Prediction button
if st.button("Predict", key="predict_btn"):
    # Convert input into DataFrame
    input_data_model = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
                                    columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    # Encoding categorical values
    mappings = {
        'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
        'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
        'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
        'transmission': {'Manual': 1, 'Automatic': 2},
        'name': {brand: idx+1 for idx, brand in enumerate(cars_data['name'].unique())}  # Dynamic encoding
    }

    for col, mapping in mappings.items():
        input_data_model[col] = input_data_model[col].map(mapping)

    # Predict the car price
    try:
        car_price = model.predict(input_data_model)
        st.markdown(f'<p class="prediction-result">Estimated Car Price: ₹ {car_price[0]:,.2f}</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

