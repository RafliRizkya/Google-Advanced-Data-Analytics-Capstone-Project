# Google Advanced Data Analytics Capstone Project

## Business Objective

Employee attrition is a critical concern for companies, especially in competitive industries where talent retention directly impacts productivity and operational costs. This project aims to build predictive models that identify employees who are likely to resign, based on factors such as satisfaction level, number of projects, average monthly hours, and tenure.

By understanding the drivers behind employee turnover, HR departments can:
- Take proactive steps to retain valuable employees.
- Identify areas for improvement such as burnout factors while establishing minimum project requirements and promotion criteria.

The ultimate goal is to support data-driven decision-making within Human Resources by leveraging machine learning models that predict attrition with high accuracy.

---

## Technical Proficiency

### 1. Exploratory Data Analysis (EDA)
Extensive data exploration was conducted to understand feature behavior and relationships. Core activities included:
- **Outlier Detection**
- **Data Cleansing**
- **Correlation Analysis & Statistical Summaries**

### 2. Machine Learning

#### Supervised Learning
- **Logistic Regression** (with regularization)
- **Naive Bayes (GaussianNB)**
- **Decision Tree**
- **Ensemble Models**:
  - **Random Forest**
  - **XGBoost**

> *GridSearchCV was used to perform hyperparameter tuning for Decision Tree, Random Forest, and XGBoost models.*

#### Unsupervised Learning (Intermediate)
- **K-Means Clustering**
  - Cluster evaluation using **Silhouette Score** and **Elbow Method**

---

## Tools & Libraries

The project was implemented in Python, using the following libraries:

- `pandas`, `numpy` – for data manipulation
- `seaborn`, `matplotlib.pyplot` – for data visualization
- `scikit-learn` – for preprocessing, modeling, evaluation, and hyperparameter tuning:
  - `train_test_split`, `GridSearchCV`
  - `StandardScaler`
  - `LogisticRegression`, `GaussianNB`, `DecisionTreeClassifier`, `RandomForestClassifier`, `KMeans`
  - `f1_score`, `recall_score`, `precision_score`, `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `classification_report`, `silhouette_score`
- `xgboost` – for gradient boosting model (`XGBClassifier`)
- `joblib` – for saving and loading trained models

---

## Key Takeaways

This capstone demonstrates a complete data analytics pipeline, from initial data exploration to building and evaluating supervised and unsupervised models. Through effective use of model tuning and statistical reasoning, the project provides practical insights that can support HR decision-making processes.

---

## License

This project was developed for educational purposes under the Google Advanced Data Analytics Professional Certificate program.
