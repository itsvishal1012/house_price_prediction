# 🏡 House Price Prediction using Machine Learning

This project is a comprehensive data science pipeline to predict house prices using machine learning models. The pipeline includes data loading, cleaning, exploratory data analysis (EDA), feature engineering, feature selection, and preparation for model training. The project is implemented in Jupyter notebooks and uses a structured approach for clarity and reproducibility.

---

## 📌 Project Goals

- Understand the structure and nature of housing datasets
- Perform in-depth exploratory data analysis (EDA)
- Clean, preprocess, and transform raw data into usable form
- Apply feature engineering to derive new insights
- Perform feature selection to reduce dimensionality and improve accuracy
- Prepare data for machine learning model training and evaluation

---

## 🗂️ Dataset Information

This project uses two primary datasets:

### `train.csv`
- Contains the main training data used for EDA, preprocessing, and modeling.
- Each row represents a house with various features such as:
  - `Location`, `Size`, `Total_Square_Feet`, `Bathroom`, `Balcony`, `BHK`
  - Target variable: `Price`

### `test.csv`
- Contains new, unseen house listings on which the model's prediction accuracy can be evaluated.
- Has the same features as `train.csv`, **except** the target `Price`.

---

## 🧪 Notebooks Summary

### 1. 📊 `Exploratory_Data_Analysis.ipynb`

**Purpose**: Explore data structure, identify issues, and understand variable relationships.

**Main Tasks:**
- Load and inspect data
- Visualize numerical distributions using histograms, box plots, violin plots
- Detect and handle missing values
- Identify and treat outliers
- Analyze correlations between numerical features using heatmaps
- Analyze relationships using pair plots, scatter plots, and categorical plots (countplots, barplots)

---

### 2. 🛠️ `Feature_Engineering.ipynb`

**Purpose**: Create or transform features to make them more useful for the model.

**Key Operations:**
- **Missing Value Imputation**: Fill null values using mean/median/mode or domain-based logic
- **Categorical Encoding**: 
  - Label encoding for ordinal variables
  - One-hot encoding for nominal variables
- **Normalization & Scaling**: Applied where required for algorithms sensitive to feature scale
- **New Feature Creation**: Combine or transform existing features to generate:
  - `Total_Square_Feet_Per_Room`, `Price_Per_Sqft`, `BHK_to_Bathroom_Ratio`, etc.
- **Skewness Handling**: Apply log/sqrt transformations where feature distributions are skewed

---

### 3. 🧠 `Feature_Selection.ipynb`

**Purpose**: Identify and retain the most relevant features to improve model accuracy and reduce overfitting.

**Techniques Applied:**
- **Correlation Matrix Analysis**: Remove multicollinear features
- **Univariate Feature Selection**: SelectKBest with statistical tests
- **Model-Based Selection**:
  - Feature importance from `RandomForestRegressor`, `XGBoost`
  - Recursive Feature Elimination (RFE)
- **Dimensionality Reduction (Optional)**: PCA or TruncatedSVD if applicable

---

### | Library             | Purpose                                                   |
| `pandas`                | Data manipulation                                         |
| `numpy`                 | Numerical computations                                    |
| `matplotlib`, `seaborn` | Data visualization                                        |
| `scikit-learn`          | Machine learning models, preprocessing, feature selection |
| `xgboost` / `lightgbm`  | Advanced tree-based algorithms (optional, for modeling)   |

### 📈 Results (To be added after modeling)
After feature processing, the dataset is clean, well-structured, and ready for training using models like:

Linear Regression

Random Forest

XGBoost

Gradient Boosting

SVR, etc.

Accuracy metrics like RMSE, MAE, and R² Score will be used to evaluate model performance.

### 🚧 Future Enhancements
Add machine learning model training and hyperparameter tuning

Use cross-validation for more robust results

Deploy the model using Flask/Streamlit

Create an interactive web dashboard for predictions

Automate the pipeline using scripts or MLflow


## 🖥️ How to Run This Project

1. **Clone the Repository**

```bash
git clone https://github.com/itsvishal1012/house_price_prediction.git
cd house_price_prediction
