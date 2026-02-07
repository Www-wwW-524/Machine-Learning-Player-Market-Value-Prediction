# Machine-Learning-Player-Market-Value-Prediction
This project focuses on predicting professional football players’ market value using tabular data.
The objective is to build a **robust end-to-end machine learning pipeline**, from exploratory data analysis and preprocessing to model comparison and evaluation, with particular attention to data structure, noise, and multicollinearity.

---

## Data
- **Observations:** 15,391 players  
- **Features:** 75 player attributes (technical, physical, demographic)
- **Target variable:** Player market value (EUR)

Exploratory analysis revealed **role-driven feature availability**, with systematic differences between goalkeepers and outfield players.

---

## Data Preprocessing
Key preprocessing steps included:
- Role-specific feature handling for goalkeepers vs. outfield players
- Removal of high-cardinality identifiers
- Median imputation and encoding of categorical variables
- Log transformation of market value to address heavy skewness

The log transformation reduced target skewness from **7.8 to 0.57** and kurtosis from **84 to 0.84**, significantly improving model stability.

---

## Models
The following models were implemented and compared:
- **OLS and Ridge Regression** as linear baselines
- **Random Forest** to capture nonlinear feature interactions
- **XGBoost** as a boosting-based ensemble model

Models were evaluated using **5-fold cross-validation** with **RMSE** on both log-transformed and original value scales.

---

## Results
- Ensemble methods outperformed linear baselines on predictive accuracy
- Random Forest achieved the most **stable performance**, reducing RMSE on the original value scale from approximately **€628k to €490k**
- Boosting models required careful tuning and showed higher sensitivity to noise and correlated features
