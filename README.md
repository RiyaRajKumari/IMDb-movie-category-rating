# IMDb Movie Rating Predictor

An MSc Big Data Analytics project that predicts whether a movie is likely to receive an IMDb rating of 7.0 or higher using an XGBoost classification model.

## 🚀 Live Demo

**[Click here to open the IMDb Movie Rating Predictor](https://imdb-movie-category-rating-nhkqeakaqyqe7edjwwkd6x.streamlit.app/)**

## Features

- Predicts the high-rated IMDb category (rating 7.0+).
- Accepts movie name, release year, runtime, adult-content status, and genres.
- Displays a model score and predicted category.
- Built with Python, XGBoost, and Streamlit.

## Model Information

- **Algorithm:** XGBoost
- **Test accuracy:** 73.62%
- **Classification threshold:** 0.41

The model predicts a rating category, not the exact IMDb rating. Test accuracy is an evaluation result and does not guarantee an individual movie's prediction.

## Technologies Used

Python, Streamlit, Pandas, Scikit-learn, XGBoost, Joblib.
