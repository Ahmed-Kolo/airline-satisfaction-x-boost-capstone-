# README: Airline Customer Satisfaction Prediction with XGBoost

## Repository Name
`airline-satisfaction-xgboost`

## Description (less than 300 characters)
Predicting airline customer satisfaction using XGBoost. Explores data preprocessing, hyperparameter tuning with GridSearchCV, comparative analysis, and actionable insights for service improvement. Includes dummy data generation.

## Project Overview
This Capstone project focuses on building and evaluating a robust machine learning model to predict airline customer satisfaction. Given the absence of a specific dataset, a synthetic dataset was generated to simulate realistic airline customer feedback. The core objective was to implement an XGBoost classifier, optimize its performance using `GridSearchCV`, and compare it against other tree-based models like Decision Tree and Random Forest. Ultimately, the project aims to extract actionable insights from feature importance to provide business recommendations for enhancing customer experience.

## Methodology: Milestone Breakdown

### Milestone 1: Discovery & Advanced Preprocessing
*   **Dummy Data Generation**: Created a synthetic dataset of 1000 samples, encompassing various numerical (e.g., age, flight distance, service ratings) and categorical features (e.g., gender, customer type, travel type, class), with a binary target variable `satisfaction`.
*   **Feature Identification**: Distinguished between numerical and categorical features within the dataset.
*   **Preprocessing Pipeline**: Implemented `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical features to handle data transformation effectively.
*   **Train-Test Split**: Divided the preprocessed data into training and testing sets (80% train, 20% test) to ensure robust model evaluation.

### Milestone 2: Model Construction & Hyperparameter Tuning
*   **XGBoost Classifier Initialization**: Initialized an `XGBClassifier` with `objective='binary:logistic'` and `eval_metric='logloss'`.
*   **Hyperparameter Grid**: Defined a comprehensive parameter grid for `GridSearchCV`, including `n_estimators`, `learning_rate`, `max_depth`, `colsample_bytree`, and `subsample`.
*   **`GridSearchCV` Implementation**: Utilized `GridSearchCV` with 3-fold cross-validation and `accuracy` scoring to systematically search for the optimal combination of hyperparameters. The process execution time was tracked using `%%time`.
*   **Best Model Selection**: Identified and stored the best-performing XGBoost model based on the grid search results.

### Milestone 3: Model Evaluation
*   **Prediction**: Used the `best_xgb_model` to make predictions on the unseen test set.
*   **Performance Metrics**: Calculated and reported standard classification metrics: Accuracy, Precision, Recall, and F1-score.
*   **Confusion Matrix**: Visualized the model's performance using a heatmap of the Confusion Matrix to understand true positives, true negatives, false positives, and false negatives.

### Milestone 4: Comparative Analysis
*   **Baseline Models**: Trained and evaluated a `DecisionTreeClassifier` and a `RandomForestClassifier` on the preprocessed training data.
*   **Metric Comparison**: Calculated Accuracy, Precision, Recall, and F1-score for both Decision Tree and Random Forest models on their respective test sets.
*   **Summary Table**: Created a comparative table summarizing the performance metrics of XGBoost, Decision Tree, and Random Forest models, providing a clear overview of their relative strengths.

### Milestone 5: Actionable Insights & Feature Importance
*   **Feature Importance Visualization**: Used `plot_importance` from the XGBoost library to visualize the most impactful features (by weight) in the `best_xgb_model`.
*   **Feature Mapping**: Mapped the internal feature names (f0, f1, etc.) back to their original descriptive names for clarity.
*   **Business Recommendations**: Provided concrete business recommendations based on the identified top features, advising on areas such as inflight service quality, check-in and baggage processes, delay management, customer segmentation, and age-group specific monitoring to enhance customer satisfaction.

## Key Findings
*   The XGBoost model achieved superior performance (e.g., Accuracy ~0.765, F1-score ~0.773) compared to the Decision Tree (Accuracy ~0.550) and Random Forest (Accuracy ~0.750) models, demonstrating its effectiveness for this classification task.
*   Critical features influencing customer satisfaction include `Flight Distance`, `Age`, `Inflight entertainment`, `Arrival Delay in Minutes`, and `Food and drink`, among others.

## Technologies Used
*   Python 3.x
*   `pandas` (for data manipulation)
*   `numpy` (for numerical operations)
*   `scikit-learn` (for preprocessing, model selection, and evaluation)
*   `xgboost` (for the gradient boosting model and feature importance)
*   `matplotlib` and `seaborn` (for data visualization)

## Usage
To replicate the analysis:
1.  Ensure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`).
2.  Run the notebook cells sequentially to execute the data generation, preprocessing, model training, evaluation, and analysis steps.
