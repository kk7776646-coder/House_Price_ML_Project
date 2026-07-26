import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.write("Streamlit is working!")

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL AND FILES
# --------------------------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("house_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    default_features = joblib.load("default_features.pkl")

    return model, scaler, feature_names, default_features


model, scaler, feature_names, default_features = load_files()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below to estimate the house sale price."
)

st.divider()


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

st.subheader("🏡 Property Details")

col1, col2, col3 = st.columns(3)


with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    total_bsmt_sf = st.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000
    )


with col2:

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500
    )

    first_flr_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=100,
        max_value=5000,
        value=1000
    )

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=2000,
        value=400
    )

    garage_cars = st.number_input(
        "Garage Cars",
        min_value=0,
        max_value=5,
        value=2
    )


with col3:

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2
    )

    half_bath = st.number_input(
        "Half Bathrooms",
        min_value=0,
        max_value=3,
        value=1
    )

    bedrooms = st.number_input(
        "Bedrooms Above Ground",
        min_value=0,
        max_value=10,
        value=3
    )

    rooms = st.number_input(
        "Total Rooms Above Ground",
        min_value=1,
        max_value=20,
        value=6
    )


# --------------------------------------------------
# CREATE INPUT DATA
# --------------------------------------------------

if st.button("🔮 Predict House Price", type="primary"):

    # Convert default feature Series into DataFrame
    input_data = pd.DataFrame(
        [default_features.values],
        columns=default_features.index
    )

    # --------------------------------------------------
    # Replace important numerical features
    # --------------------------------------------------

    values_to_update = {
        "OverallQual": overall_qual,
        "YearBuilt": str(year_built),
        "YearRemodAdd": str(year_remod),
        "TotalBsmtSF": total_bsmt_sf,
        "GrLivArea": gr_liv_area,
        "1stFlrSF": first_flr_sf,
        "GarageArea": garage_area,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": rooms
    }


    # --------------------------------------------------
    # Update only features that actually exist
    # --------------------------------------------------

    for feature, value in values_to_update.items():

        if feature in input_data.columns:
            input_data[feature] = value


    # --------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )


    # --------------------------------------------------
    # SCALE INPUT DATA
    # --------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = model.predict(input_scaled)[0]


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.success("Prediction completed successfully!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.2f}"
    )

    st.info(
        "This is an estimated price generated by the trained "
        "Gradient Boosting Regressor model."
    )
