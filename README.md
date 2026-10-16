# IMDb Movie Rating Category Prediction

An end-to-end Big Data Analytics and Machine Learning project using IMDb movie metadata to analyse rating patterns and predict whether a movie belongs to the IMDb 7+ rating category.

## Project Overview

This project explores movie data through data preparation, exploratory data analysis (EDA), statistical analysis, machine learning and interactive visualisation.

The project includes a machine-learning predictor deployed with Streamlit and a separate interactive IMDb analytics dashboard.

The goal is to demonstrate how data analytics can support the exploration of entertainment trends and audience insights.

## Live Applications

- **IMDb movie category rating:** https://imdb-movie-category-rating-nhkqeakaqyqe7edjwwkd6x.streamlit.app/
- **Interactive IMDb Dashboard:** https://imdb-dashboard-mom.streamlit.app/

*Note: Please verify the predictor URL against your browser address bar before publishing.*

## Project Objectives

- Process and analyse large IMDb datasets.
- Explore rating distributions, movie genres and relationships between variables.
- Apply statistical methods to investigate differences in ratings.
- Train and evaluate machine-learning classification models.
- Deploy an interactive movie rating category predictor.
- Present analytical findings through an interactive dashboard.
- Discuss how entertainment analytics can inform audience research and marketing decisions.

## Dataset

The project uses IMDb datasets containing movie information such as:

- Title identifiers and movie titles
- Release year and runtime
- Genres
- Average IMDb ratings
- Number of votes
- Other title metadata used during data preparation

The prepared master dataset contained approximately 12.75 million rows before selecting the movie subset for the analysis.

Large source files were processed using a combination of Python, DuckDB and Pandas.

**Data note:** The raw IMDb data files are not included in this repository because of their size. Refer to IMDb's official datasets and terms for access and usage information.

## Methodology

### 1. Data Collection and Preparation

- Obtained IMDb datasets.
- Combined relevant data into a prepared dataset.
- Used DuckDB and Pandas for data processing.
- Selected movie records relevant to the research objective.
- Prepared the features and target variable for analysis.

### 2. Exploratory Data Analysis

Explored movie rating distributions, runtime, genres and other relevant variables to identify patterns and understand the data.

### 3. Statistical Analysis

- Examined differences in average ratings across genres.
- Used ANOVA to test for differences in group means.
- Used correlation analysis to investigate relationships between ratings and vote counts.

Statistical significance indicates evidence of differences or associations; it does not establish causation.

### 4. Machine Learning

The project compared multiple classification models, including:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the final model based on the evaluation results obtained during the project.

### 5. Model Evaluation

The final tuned XGBoost classifier achieved **73.62% accuracy** on the held-out temporal test set.

The model predicts a category rather than an exact IMDb rating.

### 6. Deployment and Visualisation

- Deployed the rating category predictor using Streamlit.
- Saved the trained model and supporting configuration artifacts.
- Developed a separate interactive dashboard for exploring IMDb data and trends.

## Prediction Target

The target variable is based on the IMDb average rating:

- **High-rated category:** IMDb rating greater than or equal to 7.0
- **Below-7 category:** IMDb rating below 7.0

The model uses features such as release year, runtime, adult-content flag and genre indicators.

The movie name is a display label only and is not used as a model feature.

The saved decision threshold of 0.41 is used to convert the model's output score into a predicted category. This is different from the 7.0 IMDb rating threshold used to define the target labels.

The displayed model score should not be interpreted as a calibrated probability.

## Key Findings

- Average movie ratings and rating distributions were explored across genres.
- ANOVA identified statistically significant differences in genre-level mean ratings in the analysed sample.
- The relationship between rating and vote count was weak in the correlation analysis.
- XGBoost achieved 73.62% accuracy on the held-out temporal test set.

These findings describe the analysed data and model evaluation. They do not establish that a particular genre causes higher ratings or that a high-rated movie will be commercially successful.

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data processing | Pandas, DuckDB |
| Statistical analysis | Python statistical tools |
| Machine learning | Scikit-learn, XGBoost |
| Model persistence | Joblib |
| Interactive applications | Streamlit |
| Visualisation | Plotly and other project visualisation tools, where implemented |
| Version control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

## Marketing and Entertainment Analytics Applications

The project provides a starting point for exploring how movie metadata can support entertainment research.

Related marketing frameworks include:

- **STP:** Segmentation, Targeting and Positioning
- **AIDA:** Attention, Interest, Desire and Action
- **RACE:** Reach, Act, Convert and Engage
- **4Ps:** Product, Price, Place and Promotion
- **Marketing Mix Modeling:** Estimating marketing-channel contributions using campaign and outcome data
- **Attribution and Incrementality:** Evaluating campaign touchpoints and estimating the impact of marketing activity

The current IMDb model does not measure marketing ROI, predict box-office revenue or determine campaign effectiveness. Those tasks would require additional data, such as campaign expenditure, audience exposure, ticket sales or streaming outcomes.

## Repository Structure

```text
IMDb-movie-category-rating/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── model_features.joblib
│   ├── xgboost_config.joblib
│   ├── xgboost_model.joblib
│   └── xgboost_threshold.joblib
├── analysis/
├── dashboard/
├── presentation/
└── docs/
```

The `analysis/`, `dashboard/`, `presentation/` and `docs/` folders can be added as the corresponding project files are organised. The model artifacts listed above should remain in `models/` if those are their actual repository paths.

## Limitations

- The model predicts a rating category, not an exact rating.
- IMDb ratings may change over time.
- The model uses a limited set of movie metadata features.
- Model performance may vary on newer or different types of movies.
- Rating category is not equivalent to audience size, box-office revenue or marketing success.
- Further validation and additional data would be required for commercial decision-making.

## Future Improvements

- Expand the interactive dashboard with genre filters, rating distributions and release-year trends.
- Add review sentiment analysis if suitable text data is available.
- Investigate recommendation or clustering approaches.
- Explore box-office or streaming outcomes if reliable data is available.
- Improve model monitoring, calibration and evaluation on newer data.

## Academic Context

This project was developed as part of an MSc Big Data Analytics project to demonstrate an end-to-end workflow involving data preparation, statistical analysis, machine learning and deployment.

## Acknowledgements

Data source: IMDb datasets. Please consult IMDb's official dataset documentation and applicable terms of use.

## Author

**Riya Raj Kumari**

MSc Big Data Analytics
