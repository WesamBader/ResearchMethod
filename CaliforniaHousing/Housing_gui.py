
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------------
# WEBSITE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="Home Value Lab",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Home Value Lab")
st.write("Explore home features and predict California neighborhood housing values.")
st.caption(
    "Educational project. Estimates use historical California "
    "neighborhood data and are not current home appraisals."
)

MODEL_FILE = Path(__file__).resolve().parent / "housing_web_model.joblib"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]


# -----------------------------------
# TRAIN OR LOAD MODEL
# -----------------------------------

@st.cache_resource(show_spinner="Loading housing prediction model...")
def get_model():
    if MODEL_FILE.exists():
        try:
            saved = joblib.load(MODEL_FILE)
            if (
                saved.get("features") == FEATURES
                and saved.get("version") == 1
            ):
                return saved
        except Exception:
            pass

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df[FEATURES]
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    saved = {
        "version": 1,
        "model": model,
        "features": FEATURES,
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions) * 100000
    }

    try:
        joblib.dump(saved, MODEL_FILE)
    except OSError:
        pass

    return saved


try:
    saved = get_model()
    model = saved["model"]
except Exception as error:
    st.error(f"Could not load or train the model: {error}")
    st.info("The first run requires an internet connection.")
    st.stop()


# -----------------------------------
# WEBSITE TABS
# -----------------------------------

tab1, tab2, tab3 = st.tabs([
    "🏡 Price Predictor",
    "📊 Housing Charts",
    "🏠 Home Details"
])


# -----------------------------------
# TAB 1: PRICE PREDICTOR
# -----------------------------------

with tab1:

    st.header("California Housing Price Predictor")

    st.info(
        "Enter neighborhood information below. "
        "This predicts historical median housing value "
        "for a California census block group."
    )

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Median Income ($10,000 units)",
            min_value=0.1,
            max_value=15.0,
            value=4.0,
            step=0.1
        )

        age = st.number_input(
            "Median House Age",
            min_value=1.0,
            max_value=52.0,
            value=25.0
        )

        rooms = st.number_input(
            "Average Rooms per Household",
            min_value=0.1,
            value=5.5
        )

        bedrooms = st.number_input(
            "Average Bedrooms per Household",
            min_value=0.1,
            value=1.1
        )

    with col2:
        population = st.number_input(
            "Neighborhood Population",
            min_value=1.0,
            value=1200.0
        )

        occupancy = st.number_input(
            "Average People per Household",
            min_value=0.1,
            value=3.0
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.1,
            value=34.05
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-124.5,
            max_value=-114.0,
            value=-118.25
        )

    neighborhood = {
        "MedInc": income,
        "HouseAge": age,
        "AveRooms": rooms,
        "AveBedrms": bedrooms,
        "Population": population,
        "AveOccup": occupancy,
        "Latitude": latitude,
        "Longitude": longitude
    }

    input_data = pd.DataFrame([neighborhood])[FEATURES]

    if st.button("🏡 Predict Housing Price", type="primary"):

        try:
            prediction = float(model.predict(input_data)[0])
            price = prediction * 100000

            st.session_state["predicted_price"] = price

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if "predicted_price" in st.session_state:

        price = st.session_state["predicted_price"]

        st.success("Prediction Complete!")

        st.metric(
            "Estimated Median Housing Value",
            f"${price:,.0f}"
        )

        st.caption(
            "This estimate is based on historical data, "
            "not a current property listing."
        )

    st.subheader("Model Performance")

    col3, col4 = st.columns(2)

    col3.metric(
        "R² Score",
        f"{saved['r2']:.3f}"
    )

    col4.metric(
        "Mean Absolute Error",
        f"${saved['mae']:,.0f}"
    )


# -----------------------------------
# TAB 2: HOUSING CHARTS
# -----------------------------------

with tab2:

    st.header("Interactive Housing Charts")

    st.write(
        "Explore how different neighborhood features "
        "affect the model's estimated housing values."
    )

    chart_type = st.selectbox(
        "Choose a chart",
        [
            "Median Income vs Housing Value",
            "House Age vs Housing Value",
            "Average Rooms vs Housing Value"
        ]
    )

    if st.button("📊 Generate Chart"):

        if chart_type == "Median Income vs Housing Value":
            feature = "MedInc"
            values = np.linspace(1, 12, 20)
            label = "Median Income ($10,000 units)"

        elif chart_type == "House Age vs Housing Value":
            feature = "HouseAge"
            values = np.linspace(1, 52, 20)
            label = "Median House Age"

        else:
            feature = "AveRooms"
            values = np.linspace(2, 10, 20)
            label = "Average Rooms"

        rows = []

        for value in values:
            example = neighborhood.copy()
            example[feature] = float(value)
            rows.append(example)

        chart_inputs = pd.DataFrame(rows)[FEATURES]

        chart_predictions = (
            model.predict(chart_inputs) * 100000
        )

        chart_data = pd.DataFrame({
            label: values,
            "Estimated Value ($)": chart_predictions
        }).set_index(label)

        st.line_chart(chart_data)

        st.dataframe(
            chart_data.style.format({
                "Estimated Value ($)": "${:,.0f}"
            }),
            use_container_width=True
        )

        st.caption(
            "The chart changes one neighborhood feature "
            "at a time while keeping the others constant."
        )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance.set_index("Feature")
    )


# -----------------------------------
# TAB 3: INDIVIDUAL HOME DETAILS
# -----------------------------------

with tab3:

    st.header("Individual Home Information")

    st.write(
        "Enter details about a home you are interested in. "
        "These fields can be used with a future model "
        "trained on actual home sales."
    )

    col5, col6 = st.columns(2)

    with col5:
        home_bedrooms = st.number_input(
            "Bedrooms", 0, 20, 3
        )

        bathrooms = st.number_input(
            "Bathrooms", 0.0, 20.0, 2.0, step=0.5
        )

        square_feet = st.number_input(
            "Living Area (Square Feet)",
            100, 30000, 1800
        )

        garage = st.number_input(
            "Garage Spaces", 0, 10, 2
        )

    with col6:
        lot_size = st.number_input(
            "Lot Size (Square Feet)",
            0, 1000000, 6000
        )

        year_built = st.number_input(
            "Year Built", 1700, 2026, 2000
        )

        city = st.text_input(
            "City", "Los Angeles"
        )

        zip_code = st.text_input(
            "ZIP Code", "90001"
        )

        property_type = st.selectbox(
            "Property Type",
            [
                "Single Family",
                "Condo",
                "Townhouse",
                "Multi Family"
            ]
        )

    home_data = pd.DataFrame([{
        "Bedrooms": home_bedrooms,
        "Bathrooms": bathrooms,
        "Square Feet": square_feet,
        "Garage Spaces": garage,
        "Lot Size": lot_size,
        "Year Built": year_built,
        "City": city,
        "ZIP Code": zip_code,
        "Property Type": property_type
    }])

    st.subheader("Your Home Details")

    st.dataframe(
        home_data,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "An individual-home price prediction is not yet "
        "available. The California dataset does not include "
        "these property-level features. A separate dataset "
        "of sold homes is needed to train this model."
    )

    st.download_button(
        "Download Home Information",
        data=home_data.to_csv(index=False),
        file_name="home_information.csv",
        mime="text/csv"
    )


st.divider()

st.caption(
    "Home Value Lab | Chapter 16 Regression Project"
)
