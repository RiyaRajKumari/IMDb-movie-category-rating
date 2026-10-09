
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="IMDb Movie Rating Predictor",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load saved model artifacts
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_DIR / "xgboost_model.joblib")
    threshold = joblib.load(MODEL_DIR / "xgboost_threshold.joblib")
    features = joblib.load(MODEL_DIR / "model_features.joblib")
    return model, float(threshold), features

try:
    model, threshold, feature_names = load_model_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model files: {exc}")
    st.stop()

# -----------------------------
# App heading
# -----------------------------
st.title("🎬 IMDb Movie Rating Predictor")
st.write(
    "Predict whether a movie is likely to receive an IMDb rating "
    "of 7.0 or higher using its release year, runtime, adult status, "
    "and genres."
)

st.info(
    f"Model: XGBoost | Decision threshold: {threshold:.2f}"
)

# -----------------------------
# User inputs
# -----------------------------
st.sidebar.header("Movie Details")

current_year = pd.Timestamp.now().year
movie_name = st.sidebar.text_input(
    "Movie Name",
    placeholder="e.g., Inception"
)
release_year = st.sidebar.number_input(
    "Release year",
    min_value=1870,
    max_value=current_year + 5,
    value=2020,
    step=1
)

runtime = st.sidebar.number_input(
    "Runtime (minutes)",
    min_value=1,
    max_value=600,
    value=100,
    step=1
)

is_adult = st.sidebar.selectbox(
    "Adult content",
    options=["No", "Yes"]
)

available_genres = [
    "Action", "Adult", "Adventure", "Animation", "Biography",
    "Comedy", "Crime", "Documentary", "Drama", "Family",
    "Fantasy", "Film-Noir", "Game-Show", "History", "Horror",
    "Music", "Musical", "Mystery", "News", "Reality-TV",
    "Romance", "Sci-Fi", "Sport", "Talk-Show", "Thriller",
    "War", "Western"
]

selected_genres = st.sidebar.multiselect(
    "Select genres",
    options=available_genres,
    default=["Drama"]
)

# -----------------------------
# Prepare features in training order
# -----------------------------
input_values = {
    "startYear": release_year,
    "runtimeMinutes": float(runtime),
    "isAdult": int(is_adult == "Yes")
}

for genre in available_genres:
    input_values[f"genre_{genre}"] = int(genre in selected_genres)

input_df = pd.DataFrame(
    [[input_values.get(feature, 0) for feature in feature_names]],
    columns=feature_names
)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict IMDb Rating Category", type="primary"):
    try:
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= threshold)

        st.subheader(
            f"Prediction for: {movie_name.strip()}"
            if movie_name.strip()
            else "Movie Prediction"
        )

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("Predicted category: High-rated (IMDb 7+)")
            else:
                st.warning("Predicted category: Below IMDb 7")

        with col2:
            st.metric(
                "Model score for high-rated category",
                f"{probability:.1%}"
            )

        st.progress(min(max(probability, 0.0), 1.0))

        st.caption(
            f"The model classifies a movie as high-rated when its score "
            f"is at least {threshold:.2f}. The score is not the movie's actual "
            "IMDb rating and should not be interpreted as a calibrated probability."
        )

        with st.expander("View input features"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# -----------------------------
# Model information
# -----------------------------
st.divider()
st.subheader("About This Model")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "XGBoost")
col2.metric("Test accuracy", "73.62%")
col3.metric("Classification threshold", f"{threshold:.2f}")

st.write(
    "The model was evaluated on a temporally separated test set of newer "
    "movies. Its reported test accuracy is an evaluation result, not a "
    "guarantee for an individual movie."
)

st.caption(
    "Academic project prototype. IMDb ratings can change over time, and "
    "predictions should not be treated as actual IMDb ratings."
)
