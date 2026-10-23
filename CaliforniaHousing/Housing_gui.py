
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# WEBSITE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Home Value Lab",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Home Value Lab")
st.write("California Housing Price Prediction Using Machine Learning")
st.caption(
    "Educational project using historical California neighborhood housing data."
)

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

MODEL_PATH = (
    Path(__file__).resolve().parent / "housing_web_model.joblib"
)


# ==========================================
# TRAIN MODEL WITH 5-FOLD CROSS-VALIDATION
# ==========================================

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)

            if (
                saved.get("version") == 2
                and saved.get("features") == FEATURES
            ):
                return saved

        except Exception:
            pass

    # Download California housing dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df[FEATURES]
    y = df["MedHouseVal"]

    # Separate training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest Regression Model
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # 5-Fold Cross-Validation
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1
    )

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # Train final model on full training set
    model.fit(X_train, y_train)

    # Evaluate on untouched test set
    test_predictions = model.predict(X_test)

    r2 = r2_score(y_test, test_predictions)

    mae = (
        mean_absolute_error(y_test, test_predictions)
        * 100000
    )

    saved = {
        "version": 2,
        "features": FEATURES,
        "model": model,
        "r2": float(r2),
        "mae": float(mae),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": cv_mean,
        "cv_std": cv_std
    }

    # Save model so it does not retrain every time
    try:
        joblib.dump(saved, MODEL_PATH)
    except OSError:
        pass

    return saved


with st.spinner(
    "Loading model and running 5-fold cross-validation..."
):
    try:
        saved = load_model()
        model = saved["model"]

    except Exception as error:
        st.error(f"Model error: {error}")
        st.info(
            "Check your internet connection and installed libraries."
        )
        st.stop()


# ==========================================
# WEBSITE TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏡 Price Predictor",
    "📊 Housing Charts",
    "🔬 Cross-Validation",
    "🏠 Home Details"
])


# ==========================================
# TAB 1: PRICE PREDICTOR
# ==========================================

with tab1:

    st.header("California Housing Price Predictor")

    st.info(
        "This predicts historical median housing values "
        "for California neighborhoods, not individual homes."
    )

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Median Income ($10,000 units)",
            min_value=0.1,
            max_value=15.0,
            value=4.0
        )

        age = st.number_input(
            "Median House Age",
            min_value=1.0,
            max_value=52.0,
            value=25.0
        )

        rooms = st.number_input(
            "Average Rooms",
            min_value=0.1,
            value=5.5
        )

        bedrooms = st.number_input(
            "Average Bedrooms",
            min_value=0.1,
            value=1.1
        )

    with col2:

        population = st.number_input(
            "Population",
            min_value=1.0,
            value=1200.0
        )

        occupancy = st.number_input(
            "Average Occupancy",
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

    if st.button(
        "🏡 Predict Housing Price",
        type="primary"
    ):

        prediction = float(
            model.predict(input_data)[0]
        )

        st.session_state["predicted_price"] = (
            prediction * 100000
        )

    if "predicted_price" in st.session_state:

        price = st.session_state["predicted_price"]

        st.success("Prediction Complete!")

        st.metric(
            "Estimated Median Housing Value",
            f"${price:,.0f}"
        )

    st.subheader("Model Performance")

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Test R² Score",
        f"{saved['r2']:.3f}"
    )

    metric2.metric(
        "Test Mean Absolute Error",
        f"${saved['mae']:,.0f}"
    )


# ==========================================
# TAB 2: HOUSING CHARTS
# ==========================================

with tab2:

    st.header("Interactive Housing Charts")

    chart_type = st.selectbox(
        "Choose a Chart",
        [
            "Income vs Housing Value",
            "House Age vs Housing Value",
            "Rooms vs Housing Value"
        ]
    )

    if st.button("📊 Generate Chart"):

        if chart_type == "Income vs Housing Value":

            feature = "MedInc"
            values = np.linspace(1, 12, 20)
            label = "Median Income ($10,000 units)"

        elif chart_type == "House Age vs Housing Value":

            feature = "HouseAge"
            values = np.linspace(1, 52, 20)
            label = "House Age"

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

        predictions = (
            model.predict(chart_inputs) * 100000
        )

        chart_data = pd.DataFrame({
            label: values,
            "Estimated Price ($)": predictions
        }).set_index(label)

        st.line_chart(chart_data)

        st.dataframe(
            chart_data,
            use_container_width=True
        )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.dataframe(
        importance,
        hide_index=True
    )


# ==========================================
# TAB 3: 5-FOLD CROSS-VALIDATION
# ==========================================

with tab3:

    st.header("5-Fold Cross-Validation")

    st.write(
        "The training dataset is divided into five folds. "
        "The model trains on four folds and validates on "
        "the remaining fold. This process repeats five times."
    )

    st.subheader("Cross-Validation Summary")

    cv_col1, cv_col2 = st.columns(2)

    cv_col1.metric(
        "Mean Cross-Validation R²",
        f"{saved['cv_mean']:.4f}"
    )

    cv_col2.metric(
        "Standard Deviation",
        f"{saved['cv_std']:.4f}"
    )

    cv_results = pd.DataFrame({
        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],
        "R² Score": saved["cv_scores"]
    })

    st.subheader("Individual Fold Results")

    st.dataframe(
        cv_results,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Cross-Validation Bar Chart")

    st.bar_chart(
        cv_results.set_index("Fold")
    )

    st.subheader("Interpretation")

    st.write(
        f"The average R² score across the five folds "
        f"was {saved['cv_mean']:.4f}. "
        f"The standard deviation was "
        f"{saved['cv_std']:.4f}. "
        "A higher R² score indicates better predictive "
        "performance. A lower standard deviation suggests "
        "more consistent results across the folds."
    )

    st.info(
        "Cross-validation uses only the training set. "
        "The separate test set is reserved for final evaluation."
    )


# ==========================================
# TAB 4: HOME DETAILS
# ==========================================

with tab4:

    st.header("Individual Home Details")

    st.write(
        "Enter information about an individual home."
    )

    left, right = st.columns(2)

    with left:

        home_bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

        square_feet = st.number_input(
            "Living Area (Square Feet)",
            min_value=100,
            max_value=30000,
            value=1800
        )

        garage = st.number_input(
            "Garage Spaces",
            min_value=0,
            max_value=10,
            value=2
        )

    with right:

        lot_size = st.number_input(
            "Lot Size (Square Feet)",
            min_value=0,
            max_value=1000000,
            value=6000
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1700,
            max_value=2026,
            value=2000
        )

        city = st.text_input(
            "City",
            value="Los Angeles"
        )

        zip_code = st.text_input(
            "ZIP Code",
            value="90001"
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

    st.subheader("Your Home Information")

    st.dataframe(
        home_data,
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "The California Housing dataset does not contain "
        "individual property details such as garage spaces, "
        "bathrooms, or square footage. These inputs are "
        "not used by the current prediction model. "
        "A property-level dataset would be needed to "
        "predict individual home prices."
    )

    st.download_button(
        "Download Home Details",
        data=home_data.to_csv(index=False),
        file_name="home_details.csv",
        mime="text/csv"
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Home Value Lab | Chapter 16 Regression Project "
    "| Random Forest | 5-Fold Cross-Validation"
)
