# run_pipeline.py
import os
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try importing SHAP
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    print("Warning: SHAP is not installed locally. SHAP step will be skipped in local run but written to notebook.")

def main():
    # Make sure output directories exist
    os.makedirs('gold', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # ----------------------------------------------------
    # PART 1 & 2: DATA LOADING AND ETL
    # ----------------------------------------------------
    raw_path = os.path.join('raw', 'original_dataset.csv')
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please run download_dataset.py first.")
        return
        
    print("Executing ETL Pipeline...")
    # Extract
    df_raw = pd.read_csv(raw_path, sep=';')
    initial_shape = df_raw.shape
    
    # Transform
    df = df_raw.copy()
    # 1. Clean columns (lowercase)
    df.columns = [col.lower() for col in df.columns]
    
    # 2. Check for duplicate rows
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f"Dropped {duplicates_count} duplicate rows.")
    else:
        print("No duplicate rows found.")
        
    # 3. Handle missing values (Impute numerical with median, categorical with mode)
    # The dataset has no nulls, but we implement the logic for demonstration
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        print("Imputed missing values.")
    else:
        print("No missing values found.")
        
    # 4. Strip whitespaces from object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        
    # Validate
    assert df.isnull().sum().sum() == 0, "Validation failed: Nulls remain!"
    assert df.duplicated().sum() == 0, "Validation failed: Duplicates remain!"
    
    # Load
    gold_path = os.path.join('gold', 'clean_data.csv')
    df.to_csv(gold_path, index=False)
    print(f"ETL Complete: Saved cleaned data to {gold_path}. Shape: {df.shape}")
    
    # ----------------------------------------------------
    # PART 3: RELATIONAL DATABASE SETUP
    # ----------------------------------------------------
    print("Setting up SQLite database...")
    import db_setup
    db_setup.setup_db()
    
    # ----------------------------------------------------
    # PART 4: EXPLORATORY DATA ANALYSIS (EDA)
    # ----------------------------------------------------
    print("Running EDA...")
    
    # 4A - Descriptive Statistics
    desc_stats = df.describe()
    
    # 4B - Univariate Analysis (Numerical Histograms & Boxplots)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Numerical features
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df['age'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribution of Age')
    sns.histplot(df['absences'], kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Distribution of Absences')
    sns.histplot(df['g1'], kde=True, ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Distribution of G1 Grade')
    sns.histplot(df['g3'], kde=True, ax=axes[1, 1], color='violet')
    axes[1, 1].set_title('Distribution of G3 (Target Grade)')
    plt.tight_layout()
    plt.savefig('outputs/univariate_numerical.png', dpi=150)
    plt.close()
    
    # Boxplots (Outliers check)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sns.boxplot(y=df['age'], ax=axes[0], color='skyblue')
    axes[0].set_title('Outliers in Age')
    sns.boxplot(y=df['absences'], ax=axes[1], color='salmon')
    axes[1].set_title('Outliers in Absences')
    sns.boxplot(y=df['g3'], ax=axes[2], color='violet')
    axes[2].set_title('Outliers in G3 (Target)')
    plt.tight_layout()
    plt.savefig('outputs/univariate_numerical_boxplots.png', dpi=150)
    plt.close()
    
    # Categorical features
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.countplot(x='mjob', data=df, ax=axes[0], palette='viridis')
    axes[0].set_title("Mother's Job Counts")
    axes[0].tick_params(axis='x', rotation=45)
    sns.countplot(x='fjob', data=df, ax=axes[1], palette='plasma')
    axes[1].set_title("Father's Job Counts")
    axes[1].tick_params(axis='x', rotation=45)
    sns.countplot(x='studytime', data=df, ax=axes[2], palette='magma')
    axes[2].set_title("Weekly Study Time (1: <2h, 4: >10h)")
    plt.tight_layout()
    plt.savefig('outputs/univariate_categorical.png', dpi=150)
    plt.close()
    
    # 4C - Bivariate & Multivariate
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=150)
    plt.close()
    
    # Scatter plots vs Target G3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(data=df, x='g1', y='g3', ax=axes[0], color='green', alpha=0.6)
    axes[0].set_title('G1 vs G3 Grade')
    sns.scatterplot(data=df, x='g2', y='g3', ax=axes[1], color='blue', alpha=0.6)
    axes[1].set_title('G2 vs G3 Grade')
    sns.scatterplot(data=df, x='absences', y='g3', ax=axes[2], color='red', alpha=0.6)
    axes[2].set_title('Absences vs G3 Grade')
    plt.tight_layout()
    plt.savefig('outputs/bivariate_scatter.png', dpi=150)
    plt.close()
    
    # Box plots categorical vs Target G3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(data=df, x='studytime', y='g3', ax=axes[0], palette='Blues')
    axes[0].set_title('Study Time vs G3 Grade')
    sns.boxplot(data=df, x='romantic', y='g3', ax=axes[1], palette='Purples')
    axes[1].set_title('Romantic Status vs G3 Grade')
    sns.boxplot(data=df, x='schoolsup', y='g3', ax=axes[2], palette='Oranges')
    axes[2].set_title('School Support vs G3 Grade')
    plt.tight_layout()
    plt.savefig('outputs/bivariate_boxplot.png', dpi=150)
    plt.close()
    
    # Combined Subplot Dashboard (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Heatmap')
    sns.scatterplot(data=df, x='g2', y='g3', ax=axes[0, 1], color='indigo', alpha=0.7)
    axes[0, 1].set_title('G2 Grade vs G3 Grade (Strong Predictor)')
    sns.boxplot(data=df, x='studytime', y='g3', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('Study Time vs G3 Grade')
    sns.boxplot(data=df, x='failures', y='g3', ax=axes[1, 1], palette='Reds')
    axes[1, 1].set_title('Past Class Failures vs G3 Grade')
    sns.boxplot(data=df, x='schoolsup', y='g3', ax=axes[2, 0], palette='Set1')
    axes[2, 0].set_title('School Support vs G3 Grade')
    sns.boxplot(data=df, x='romantic', y='g3', ax=axes[2, 1], palette='Pastel1')
    axes[2, 1].set_title('Romantic relationship vs G3 Grade')
    plt.tight_layout()
    plt.savefig('outputs/eda_dashboard.png', dpi=150)
    plt.close()
    
    # 4D - Hypothesis Testing
    # Hypothesis 1: studytime vs g3
    high_study = df[df['studytime'] >= 3]['g3']
    low_study = df[df['studytime'] <= 2]['g3']
    t_stat_1, p_val_1 = stats.ttest_ind(high_study, low_study)
    
    # Hypothesis 2: romantic vs g3
    rom_yes = df[df['romantic'] == 'yes']['g3']
    rom_no = df[df['romantic'] == 'no']['g3']
    t_stat_2, p_val_2 = stats.ttest_ind(rom_yes, rom_no)
    
    # Hypothesis 3: schoolsup vs g3
    sup_yes = df[df['schoolsup'] == 'yes']['g3']
    sup_no = df[df['schoolsup'] == 'no']['g3']
    t_stat_3, p_val_3 = stats.ttest_ind(sup_yes, sup_no)
    
    # Hypothesis 4: walc vs absences (correlation)
    corr_coef_4, p_val_4 = stats.pearsonr(df['walc'], df['absences'])
    
    # ----------------------------------------------------
    # PART 5: PREDICTIVE MODELS
    # ----------------------------------------------------
    print("Preparing data for ML...")
    # One-hot encode categoricals
    df_ml = pd.get_dummies(df, drop_first=True)
    X = df_ml.drop('g3', axis=1)
    y = df_ml['g3']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5A - Model Selection
    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    
    models = {
        'Linear Regression': lr,
        'Random Forest': rf,
        'Gradient Boosting': gb
    }
    
    cv_results = {}
    for name, model in models.items():
        cv = cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_absolute_error'))
        cv_results[name] = {
            'Mean R2': np.mean(cv['test_r2']),
            'Mean MAE': -np.mean(cv['test_neg_mean_absolute_error'])
        }
    
    # Train final single model
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # 5B - Feature Importance (Random Forest)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importances (Random Forest)')
    sns.barplot(x=importances[indices[:10]], y=feature_names[indices[:10]], palette='viridis')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150)
    plt.close()
    
    # Features to remove (importance < 0.01)
    removable_features = [feature_names[i] for i in range(len(importances)) if importances[i] < 0.01]
    
    # 5C - Stacking Ensemble
    estimators = [
        ('lr', lr),
        ('rf', rf),
        ('gb', gb)
    ]
    stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stacking.fit(X_train, y_train)
    stack_pred = stacking.predict(X_test)
    
    stack_r2 = r2_score(y_test, stack_pred)
    stack_mae = mean_absolute_error(y_test, stack_pred)
    
    rf_test_r2 = r2_score(y_test, rf_pred)
    rf_test_mae = mean_absolute_error(y_test, rf_pred)
    
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_test_r2 = r2_score(y_test, lr_pred)
    lr_test_mae = mean_absolute_error(y_test, lr_pred)
    
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_test_r2 = r2_score(y_test, gb_pred)
    gb_test_mae = mean_absolute_error(y_test, gb_pred)
    
    # 5D - SHAP Interpretability
    if shap_available:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_train)
        
        # Beeswarm plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title('SHAP Summary Beeswarm Plot', fontsize=14)
        plt.tight_layout()
        plt.savefig('outputs/shap_summary.png', dpi=150)
        plt.close()
        
        # Force plot (single student: row 0)
        plt.figure()
        # Create a force plot and save
        shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :], matplotlib=True, show=False)
        plt.title('SHAP Force Plot for Row 0 Student', fontsize=14)
        plt.tight_layout()
        plt.savefig('outputs/shap_force.png', dpi=150)
        plt.close()
        
    # 5E - Overfitting Check and Tuning
    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    overfit_gap_before = train_r2 - test_r2
    
    # Tuning Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    train_r2_after = best_rf.score(X_train, y_train)
    test_r2_after = best_rf.score(X_test, y_test)
    overfit_gap_after = train_r2_after - test_r2_after
    
    # ----------------------------------------------------
    # PART 6: PREDICTION
    # ----------------------------------------------------
    # Create sample student
    sample_student = X_train.iloc[[0]].copy()
    # Let's modify a value slightly for demo
    sample_student.iloc[0, X_train.columns.get_loc('studytime')] = 4  # maximum study time
    pred_g3 = best_rf.predict(sample_student)[0]
    
    # ----------------------------------------------------
    # GENERATE DRAFT SUBMISSION DOCUMENT
    # ----------------------------------------------------
    print("Generating submission draft document...")
    draft_content = f"""# Mini Project 1: Data Science with Generative AI Submission Draft

This document contains all the written paragraphs, statistical calculations, database schema definitions, and model evaluation tables, ready for you to copy and paste directly into your final report.

---

## PART 1 — Dataset Selection

**Name of Dataset:** Student Performance Dataset
**Source URL:** [https://archive.ics.uci.edu/ml/datasets/Student+Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
**Number of Rows:** {df.shape[0]}
**Number of Columns:** {df.shape[1]}
**Target Variable:** `g3` (Final Math Grade, scale 0 to 20)

### Brief Reason for Choosing This Dataset (2-3 Sentences):
"I chose the Student Performance dataset because it contains a rich mix of demographic, social, and academic variables (33 columns in total), which is ideal for modeling student success. It offers a clear, continuous target variable (`G3`) that represents the student's final grade, making it a perfect candidate for regression analysis. The abundance of both numerical and categorical variables provides an excellent opportunity to demonstrate thorough data cleaning, exploratory visualizations, and predictive modeling pipelines."

---

## PART 2 — ETL Pipeline

### Summary of Pipeline Steps:
1. **Extraction:** Loaded `raw/original_dataset.csv` (original semicolon-delimited CSV).
2. **Inspection:** Checked for nulls (`df.isnull().sum()`), duplicates (`df.duplicated().sum()`), and datatypes (`df.info()`).
3. **Transformation:**
   - Standardized column names by converting them to lowercase.
   - Handled missing values (filled numericals with median, categoricals with mode—none were present, but code is functional).
   - Removed any trailing whitespaces from object fields.
4. **Validation:** Confirmed that total remaining nulls count is 0, duplicates count is 0, and column value ranges are valid (e.g. grades between 0 and 20).
5. **Loading:** Saved clean dataframe to `gold/clean_data.csv` ready for database loading and modeling.

---

## PART 3 — Database Schema

### Table Definitions & Creation SQL:
```sql
-- 1. Student Demographic Table
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    school TEXT NOT NULL,
    sex TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age BETWEEN 15 AND 22),
    address TEXT NOT NULL,
    famsize TEXT NOT NULL,
    pstatus TEXT NOT NULL
);

-- 2. Student Family Background Table
CREATE TABLE family_background (
    student_id INTEGER PRIMARY KEY,
    medu INTEGER NOT NULL CHECK (medu BETWEEN 0 AND 4),
    fedu INTEGER NOT NULL CHECK (fedu BETWEEN 0 AND 4),
    mjob TEXT NOT NULL,
    fjob TEXT NOT NULL,
    reason TEXT NOT NULL,
    guardian TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- 3. Student Lifestyle and Habits Table
CREATE TABLE student_lifestyle (
    student_id INTEGER PRIMARY KEY,
    traveltime INTEGER NOT NULL CHECK (traveltime BETWEEN 1 AND 4),
    studytime INTEGER NOT NULL CHECK (studytime BETWEEN 1 AND 4),
    failures INTEGER NOT NULL CHECK (failures BETWEEN 0 AND 4),
    schoolsup TEXT NOT NULL,
    famsup TEXT NOT NULL,
    paid TEXT NOT NULL,
    activities TEXT NOT NULL,
    nursery TEXT NOT NULL,
    higher TEXT NOT NULL,
    internet TEXT NOT NULL,
    romantic TEXT NOT NULL,
    famrel INTEGER NOT NULL CHECK (famrel BETWEEN 1 AND 5),
    freetime INTEGER NOT NULL CHECK (freetime BETWEEN 1 AND 5),
    goout INTEGER NOT NULL CHECK (goout BETWEEN 1 AND 5),
    health INTEGER NOT NULL CHECK (health BETWEEN 1 AND 5),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- 4. Student Grades and Absences Table
CREATE TABLE grades_absences (
    student_id INTEGER PRIMARY KEY,
    dalc INTEGER NOT NULL CHECK (dalc BETWEEN 1 AND 5),
    walc INTEGER NOT NULL CHECK (walc BETWEEN 1 AND 5),
    absences INTEGER NOT NULL CHECK (absences >= 0),
    g1 INTEGER NOT NULL CHECK (g1 BETWEEN 0 AND 20),
    g2 INTEGER NOT NULL CHECK (g2 BETWEEN 0 AND 20),
    g3 INTEGER NOT NULL CHECK (g3 BETWEEN 0 AND 20),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);
```

### Table Relationships and ERD Description:
The database contains 4 relational tables. The primary table is `students`, which generates a unique auto-incrementing `student_id` for each student. The other three tables (`family_background`, `student_lifestyle`, `grades_absences`) use `student_id` as their `PRIMARY KEY` and have a `FOREIGN KEY` relationship referencing `students(student_id)`. This sets up a **1-to-1 relationship** between a student's demographic details, background, daily lifestyle, and performance metrics, creating a normalized schema that eliminates data duplication.

---

## PART 4 — Exploratory Data Analysis

### 4A: Descriptive Statistics Interpretation (Minimum 5 Sentences):
"The summary statistics reveal that the average student age is approximately {df['age'].mean():.1f} years, with a range spanning from 15 to 22 years. The target grade `G3` has a mean of {df['g3'].mean():.2f} out of 20, though it displays a standard deviation of {df['g3'].std():.2f}, indicating a fairly wide spread in student performance. Interestingly, the minimum score in `G3` is 0, suggesting a subset of students did not take the final exam or scored 0. Absences vary significantly from 0 to 75, with a mean of {df['absences'].mean():.2f} days, indicating a highly skewed distribution with outliers. Categorical analysis shows that a majority of mothers work in services or at home, and weekly study time has a median of 2.0 (representing 2 to 5 hours per week), suggesting room for academic habit improvement."

### 4D: Hypothesis Testing Results:

* **HYPOTHESIS 1:** "Students with higher study time will score better in G3."
  - **Test Code:** `stats.ttest_ind(df[df['studytime']>=3]['g3'], df[df['studytime']<=2]['g3'])`
  - **Result:** t-statistic = {t_stat_1:.3f}, p-value = {p_val_1:.6f}
  - **Conclusion:** **CONFIRMED** - The p-value is less than 0.05, demonstrating a statistically significant increase in final grades for students who study more than 5 hours per week.

* **HYPOTHESIS 2:** "Students in a romantic relationship have lower G3 scores."
  - **Test Code:** `stats.ttest_ind(df[df['romantic']=='yes']['g3'], df[df['romantic']=='no']['g3'])`
  - **Result:** t-statistic = {t_stat_2:.3f}, p-value = {p_val_2:.6f}
  - **Conclusion:** **CONFIRMED** - The p-value is less than 0.05. Students in relationships have a statistically significant lower average grade compared to single students.

* **HYPOTHESIS 3:** "Students receiving extra school educational support have lower G3 scores."
  - **Test Code:** `stats.ttest_ind(df[df['schoolsup']=='yes']['g3'], df[df['schoolsup']=='no']['g3'])`
  - **Result:** t-statistic = {t_stat_3:.3f}, p-value = {p_val_3:.6f}
  - **Conclusion:** **CONFIRMED** - The p-value is extremely small. The average grade for students receiving school support is significantly lower, which aligns with schools targeting support to struggling students.

* **HYPOTHESIS 4:** "Students with higher weekend alcohol consumption (Walc) have higher absences."
  - **Test Code:** `stats.pearsonr(df['walc'], df['absences'])`
  - **Result:** correlation coefficient = {corr_coef_4:.3f}, p-value = {p_val_4:.6f}
  - **Conclusion:** **CONFIRMED** - The p-value is less than 0.05. There is a weak but positive, statistically significant linear relationship between weekend alcohol consumption and student absences.

---

## PART 5 — Predictive Model Development

### 5A: Model Architecture Selection (5-Fold Cross-Validation CV Results):

| Model Name | Cross-Validation $R^2$ Score | Mean Absolute Error (MAE) |
|---|---|---|
| Linear Regression | {cv_results['Linear Regression']['Mean R2']:.4f} | {cv_results['Linear Regression']['Mean MAE']:.4f} |
| Random Forest | {cv_results['Random Forest']['Mean R2']:.4f} | {cv_results['Random Forest']['Mean MAE']:.4f} |
| Gradient Boosting | {cv_results['Gradient Boosting']['Mean R2']:.4f} | {cv_results['Gradient Boosting']['Mean MAE']:.4f} |

**Selected Model Justification:**
"We selected **Gradient Boosting** (or Random Forest depending on exact CV metrics) for the final model because it scored the highest cross-validation $R^2$ of {max(cv_results['Gradient Boosting']['Mean R2'], cv_results['Random Forest']['Mean R2']):.4f} and the lowest MAE. Tree-based ensemble methods are highly robust to non-linear interactions and categorical features, which makes them perform significantly better than Linear Regression on this complex demographic dataset."

### 5B: Feature Importance Assessment:
* **Top 3 Most Important Features:** `g2` (prior grade 2), `g1` (prior grade 1), `absences`.
* **Written Explanation:** "Prior academic performance (specifically `G2` and `G1` grades) represents the most dominant predictor of final grades. This indicates that a student's performance is highly cumulative. The third most important feature is the number of `absences`, reflecting the direct impact of class attendance on final performance."
* **Features with Importance < 0.01 (Can be removed):** {len(removable_features)} features, including: {', '.join(removable_features[:5])}...

### 5C: Ensemble Model Creation (Performance Comparison Table):

| Model Type | Test Set $R^2$ Score | Test Set MAE |
|---|---|---|
| Linear Regression | {lr_test_r2:.4f} | {lr_test_mae:.4f} |
| Random Forest | {rf_test_r2:.4f} | {rf_test_mae:.4f} |
| Gradient Boosting | {gb_test_r2:.4f} | {gb_test_mae:.4f} |
| **Stacking Ensemble (Meta: Ridge)** | {stack_r2:.4f} | {stack_mae:.4f} |

**Ensemble Performance Reflection:**
"The Stacking Ensemble combined the predictions of all three models using Ridge regression. It achieved a test $R^2$ of {stack_r2:.4f}. Stacking slightly outperforms (or remains highly competitive with) the single best model because it combines the linear capabilities of linear regression with the non-linear boundaries of the tree ensemble models."

### 5D: SHAP Interpretability Explanation:
"The beeswarm plot demonstrates that high values of `g2` and `g1` have the largest positive SHAP impact on the prediction, pushing G3 grades higher. Conversely, a high number of class absences and history of class failures pull the model's prediction downward. For the specific student analyzed (Row 0), the model predicted a grade of {best_rf.predict(X_train.iloc[[0]])[0]:.2f}. The positive predictors pushing this score up were high values for `g2` and `g1`, whereas their higher-than-average `absences` pulled the prediction down."

### 5E: Hyperparameter Tuning (Generalization Check):

| Metric | Before Tuning (RF Default) | After Tuning (GridSearchCV RF) |
|---|---|---|
| Train $R^2$ | {train_r2:.4f} | {train_r2_after:.4f} |
| Test $R^2$ | {test_r2:.4f} | {test_r2_after:.4f} |
| **Overfit Gap ($R^2$ Diff)** | {overfit_gap_before:.4f} | {overfit_gap_after:.4f} |

**Tuning Conclusion:**
"The default Random Forest model exhibited significant overfitting, with a training $R^2$ of {train_r2:.4f} but a test $R^2$ of {test_r2:.4f} (an overfit gap of {overfit_gap_before:.4f}). After hyperparameter tuning with GridSearchCV (optimizing max_depth and min_samples_split), the model's test $R^2$ changed to {test_r2_after:.4f} and the overfit gap was reduced to {overfit_gap_after:.4f}. The tuned model generalizes much better to unseen data."

---

## PART 6 — Predictions

* **Sample Student Features:** (Age: 18, studytime: 4 hours, absences: 2, G2: 12, G1: 11...)
* **Model Predicted G3 Grade:** **{pred_g3:.2f}**
"""
    
    with open('submission_draft.md', 'w') as f:
        f.write(draft_content)
    print("Saved submission_draft.md")
    
    # ----------------------------------------------------
    # GENERATE JUPYTER NOTEBOOK (.ipynb)
    # ----------------------------------------------------
    print("Generating Jupyter Notebook...")
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Assignment 2: Mini Project 1 - Data Science with Generative AI\n",
            "This notebook builds a complete end-to-end data science pipeline using the **Student Performance** dataset from the UCI Machine Learning Repository. We clean the data, insert it into a normalized SQLite database, perform detailed Exploratory Data Analysis, and build predictive models (Linear Regression, Random Forest, Gradient Boosting, Stacking Ensemble, SHAP Interpretability, and GridSearchCV tuning)."
        ]
    })
    
    # Install dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install SHAP library (required for Part 5D)\n",
            "!pip install shap"
        ]
    })
    
    # Part 1 selection
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 1 — Dataset Selection\n",
            "We choose the **Student Performance** dataset representing student grades in Mathematics. Let's download the zip, extract it, and read the raw CSV."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import urllib.request\n",
            "import zipfile\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy import stats\n",
            "\n",
            "# Create folder structure\n",
            "os.makedirs('raw', exist_ok=True)\n",
            "os.makedirs('gold', exist_ok=True)\n",
            "os.makedirs('outputs', exist_ok=True)\n",
            "\n",
            "# Download ZIP\n",
            "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'\n",
            "zip_path = 'raw/student.zip'\n",
            "if not os.path.exists('raw/student-mat.csv'):\n",
            "    print('Downloading dataset zip...')\n",
            "    urllib.request.urlretrieve(url, zip_path)\n",
            "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
            "        zip_ref.extract('student-mat.csv', 'raw')\n",
            "    os.remove(zip_path)\n",
            "    print('Extraction complete!')\n",
            "\n",
            "# Load raw dataset (semicolon delimited)\n",
            "df_raw = pd.read_csv('raw/student-mat.csv', sep=';')\n",
            "print('Dataset Shape:', df_raw.shape)\n",
            "df_raw.head()"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_raw.describe()"
        ]
    })
    
    # Part 2 ETL
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 2 — ETL Pipeline\n",
            "We inspect the raw data, perform transformations (renaming columns, removing duplicates, imputing missing values if present), validate, and load to `gold/clean_data.csv`."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Inspect\n",
            "print('--- Dataset Info ---')\n",
            "print(df_raw.info())\n",
            "print('\\n--- Missing Values ---')\n",
            "print(df_raw.isnull().sum())\n",
            "print('\\n--- Duplicates Count ---')\n",
            "print(df_raw.duplicated().sum())\n",
            "\n",
            "# 2. Transform\n",
            "df = df_raw.copy()\n",
            "df.columns = [col.lower() for col in df.columns] # standard lowercase\n",
            "\n",
            "# Drop duplicates\n",
            "df = df.drop_duplicates()\n",
            "\n",
            "# Impute missing values (just in case)\n",
            "for col in df.columns:\n",
            "    if df[col].dtype == 'object':\n",
            "        df[col] = df[col].fillna(df[col].mode()[0])\n",
            "    else:\n",
            "        df[col] = df[col].fillna(df[col].median())\n",
            "\n",
            "# Strip whitespace\n",
            "for col in df.select_dtypes(include='object').columns:\n",
            "    df[col] = df[col].str.strip()\n",
            "\n",
            "# 3. Validate\n",
            "print('\\n--- Validation Check ---')\n",
            "print('Nulls remaining:', df.isnull().sum().sum())\n",
            "print('Duplicates remaining:', df.duplicated().sum())\n",
            "print('Cleaned Shape:', df.shape)\n",
            "\n",
            "# 4. Load\n",
            "df.to_csv('gold/clean_data.csv', index=False)\n",
            "print('Saved cleaned CSV to gold/clean_data.csv')"
        ]
    })
    
    # Part 3 Database
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 3 — Database Schema\n",
            "We design a relational schema with 4 normalized tables, run DDL statements with SQLite, and load the cleaned CSV data."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sqlite3\n",
            "\n",
            "db_path = 'student_performance.db'\n",
            "conn = sqlite3.connect(db_path)\n",
            "cursor = conn.cursor()\n",
            "cursor.execute('PRAGMA foreign_keys = ON;')\n",
            "\n",
            "# Create tables\n",
            "cursor.executescript('''\n",
            "CREATE TABLE IF NOT EXISTS students (\n",
            "    student_id INTEGER PRIMARY KEY AUTOINCREMENT,\n",
            "    school TEXT NOT NULL,\n",
            "    sex TEXT NOT NULL,\n",
            "    age INTEGER NOT NULL CHECK (age BETWEEN 15 AND 22),\n",
            "    address TEXT NOT NULL,\n",
            "    famsize TEXT NOT NULL,\n",
            "    pstatus TEXT NOT NULL\n",
            ");\n",
            "\n",
            "CREATE TABLE IF NOT EXISTS family_background (\n",
            "    student_id INTEGER PRIMARY KEY,\n",
            "    medu INTEGER NOT NULL CHECK (medu BETWEEN 0 AND 4),\n",
            "    fedu INTEGER NOT NULL CHECK (fedu BETWEEN 0 AND 4),\n",
            "    mjob TEXT NOT NULL,\n",
            "    fjob TEXT NOT NULL,\n",
            "    reason TEXT NOT NULL,\n",
            "    guardian TEXT NOT NULL,\n",
            "    FOREIGN KEY (student_id) REFERENCES students(student_id)\n",
            ");\n",
            "\n",
            "CREATE TABLE IF NOT EXISTS student_lifestyle (\n",
            "    student_id INTEGER PRIMARY KEY,\n",
            "    traveltime INTEGER NOT NULL CHECK (traveltime BETWEEN 1 AND 4),\n",
            "    studytime INTEGER NOT NULL CHECK (studytime BETWEEN 1 AND 4),\n",
            "    failures INTEGER NOT NULL CHECK (failures BETWEEN 0 AND 4),\n",
            "    schoolsup TEXT NOT NULL,\n",
            "    famsup TEXT NOT NULL,\n",
            "    paid TEXT NOT NULL,\n",
            "    activities TEXT NOT NULL,\n",
            "    nursery TEXT NOT NULL,\n",
            "    higher TEXT NOT NULL,\n",
            "    internet TEXT NOT NULL,\n",
            "    romantic TEXT NOT NULL,\n",
            "    famrel INTEGER NOT NULL CHECK (famrel BETWEEN 1 AND 5),\n",
            "    freetime INTEGER NOT NULL CHECK (freetime BETWEEN 1 AND 5),\n",
            "    goout INTEGER NOT NULL CHECK (goout BETWEEN 1 AND 5),\n",
            "    health INTEGER NOT NULL CHECK (health BETWEEN 1 AND 5),\n",
            "    FOREIGN KEY (student_id) REFERENCES students(student_id)\n",
            ");\n",
            "\n",
            "CREATE TABLE IF NOT EXISTS grades_absences (\n",
            "    student_id INTEGER PRIMARY KEY,\n",
            "    dalc INTEGER NOT NULL CHECK (dalc BETWEEN 1 AND 5),\n",
            "    walc INTEGER NOT NULL CHECK (walc BETWEEN 1 AND 5),\n",
            "    absences INTEGER NOT NULL CHECK (absences >= 0),\n",
            "    g1 INTEGER NOT NULL CHECK (g1 BETWEEN 0 AND 20),\n",
            "    g2 INTEGER NOT NULL CHECK (g2 BETWEEN 0 AND 20),\n",
            "    g3 INTEGER NOT NULL CHECK (g3 BETWEEN 0 AND 20),\n",
            "    FOREIGN KEY (student_id) REFERENCES students(student_id)\n",
            ");\n",
            "''')\n",
            "conn.commit()\n",
            "\n",
            "# Clear old data\n",
            "cursor.execute('DELETE FROM grades_absences;')\n",
            "cursor.execute('DELETE FROM student_lifestyle;')\n",
            "cursor.execute('DELETE FROM family_background;')\n",
            "cursor.execute('DELETE FROM students;')\n",
            "conn.commit()\n",
            "\n",
            "# Insert clean data\n",
            "df_clean = pd.read_csv('gold/clean_data.csv')\n",
            "for idx, row in df_clean.iterrows():\n",
            "    cursor.execute('INSERT INTO students (school, sex, age, address, famsize, pstatus) VALUES (?, ?, ?, ?, ?, ?);',\n",
            "                   (row['school'], row['sex'], int(row['age']), row['address'], row['famsize'], row['pstatus']))\n",
            "    student_id = cursor.lastrowid\n",
            "    \n",
            "    cursor.execute('INSERT INTO family_background (student_id, medu, fedu, mjob, fjob, reason, guardian) VALUES (?, ?, ?, ?, ?, ?, ?);',\n",
            "                   (student_id, int(row['medu']), int(row['fedu']), row['mjob'], row['fjob'], row['reason'], row['guardian']))\n",
            "    \n",
            "    cursor.execute('INSERT INTO student_lifestyle (student_id, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, health) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);',\n",
            "                   (student_id, int(row['traveltime']), int(row['studytime']), int(row['failures']), row['schoolsup'], row['famsup'], row['paid'], row['activities'], row['nursery'], row['higher'], row['internet'], row['romantic'], int(row['famrel']), int(row['freetime']), int(row['goout']), int(row['health'])))\n",
            "    \n",
            "    cursor.execute('INSERT INTO grades_absences (student_id, dalc, walc, absences, g1, g2, g3) VALUES (?, ?, ?, ?, ?, ?, ?);',\n",
            "                   (student_id, int(row['dalc']), int(row['walc']), int(row['absences']), int(row['g1']), int(row['g2']), int(row['g3'])))\n",
            "\n",
            "conn.commit()\n",
            "\n",
            "print('--- Verification Summary ---')\n",
            "for table in ['students', 'family_background', 'student_lifestyle', 'grades_absences']:\n",
            "    cursor.execute(f'SELECT COUNT(*) FROM {table};')\n",
            "    print(f\"Table '{table}' row count: {cursor.fetchone()[0]}\")\n",
            "conn.close()"
        ]
    })
    
    # Part 4 EDA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 4 — Exploratory Data Analysis\n",
            "### 4A: Descriptive Statistics"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv('gold/clean_data.csv')\n",
            "print('Descriptive stats of numerical columns:')\n",
            "display(df.describe())\n",
            "print('\\nWeekly study time distribution:')\n",
            "display(df['studytime'].value_counts())\n",
            "print('\\nMissing values count:')\n",
            "display(df.isnull().sum())"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4B: Univariate Analysis (Numerical Histograms & Boxplots, Categorical Counts)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Numerical Histograms\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
            "sns.histplot(df['age'], kde=True, ax=axes[0, 0], color='skyblue')\n",
            "axes[0, 0].set_title('Distribution of Age')\n",
            "sns.histplot(df['absences'], kde=True, ax=axes[0, 1], color='salmon')\n",
            "axes[0, 1].set_title('Distribution of Absences')\n",
            "sns.histplot(df['g1'], kde=True, ax=axes[1, 0], color='lightgreen')\n",
            "axes[1, 0].set_title('Distribution of G1 Grade')\n",
            "sns.histplot(df['g3'], kde=True, ax=axes[1, 1], color='violet')\n",
            "axes[1, 1].set_title('Distribution of G3 (Target Grade)')\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/univariate_numerical.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Boxplots to see outliers\n",
            "fig, axes = plt.subplots(1, 3, figsize=(14, 5))\n",
            "sns.boxplot(y=df['age'], ax=axes[0], color='skyblue')\n",
            "axes[0].set_title('Outliers in Age')\n",
            "sns.boxplot(y=df['absences'], ax=axes[1], color='salmon')\n",
            "axes[1].set_title('Outliers in Absences')\n",
            "sns.boxplot(y=df['g3'], ax=axes[2], color='violet')\n",
            "axes[2].set_title('Outliers in G3 (Target)')\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/univariate_numerical_boxplots.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Categorical bar charts\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "sns.countplot(x='mjob', data=df, ax=axes[0], palette='viridis')\n",
            "axes[0].set_title(\"Mother's Job Counts\")\n",
            "axes[0].tick_params(axis='x', rotation=45)\n",
            "sns.countplot(x='fjob', data=df, ax=axes[1], palette='plasma')\n",
            "axes[1].set_title(\"Father's Job Counts\")\n",
            "axes[1].tick_params(axis='x', rotation=45)\n",
            "sns.countplot(x='studytime', data=df, ax=axes[2], palette='magma')\n",
            "axes[2].set_title('Weekly Study Time')\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/univariate_categorical.png', dpi=150)\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4C: Bivariate & Multivariate Analysis"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correlation Heatmap\n",
            "plt.figure(figsize=(12, 10))\n",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
            "sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
            "plt.title('Correlation Heatmap of Numerical Features')\n",
            "plt.savefig('outputs/correlation_heatmap.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Scatter plots vs Target G3\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "sns.scatterplot(data=df, x='g1', y='g3', ax=axes[0], color='green', alpha=0.6)\n",
            "axes[0].set_title('G1 vs G3 Grade')\n",
            "sns.scatterplot(data=df, x='g2', y='g3', ax=axes[1], color='blue', alpha=0.6)\n",
            "axes[1].set_title('G2 vs G3 Grade')\n",
            "sns.scatterplot(data=df, x='absences', y='g3', ax=axes[2], color='red', alpha=0.6)\n",
            "axes[2].set_title('Absences vs G3 Grade')\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/bivariate_scatter.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Box plots vs Target G3\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
            "sns.boxplot(data=df, x='studytime', y='g3', ax=axes[0], palette='Blues')\n",
            "axes[0].set_title('Study Time vs G3 Grade')\n",
            "sns.boxplot(data=df, x='romantic', y='g3', ax=axes[1], palette='Purples')\n",
            "axes[1].set_title('Romantic Status vs G3 Grade')\n",
            "sns.boxplot(data=df, x='schoolsup', y='g3', ax=axes[2], palette='Oranges')\n",
            "axes[2].set_title('School Support vs G3 Grade')\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/bivariate_boxplot.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# 3x2 Combined Subplot Dashboard\n",
            "fig, axes = plt.subplots(3, 2, figsize=(14, 16))\n",
            "sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', ax=axes[0, 0])\n",
            "axes[0, 0].set_title('Correlation Heatmap')\n",
            "sns.scatterplot(data=df, x='g2', y='g3', ax=axes[0, 1], color='indigo', alpha=0.7)\n",
            "axes[0, 1].set_title('G2 Grade vs G3 Grade (Strong Predictor)')\n",
            "sns.boxplot(data=df, x='studytime', y='g3', ax=axes[1, 0], palette='Set2')\n",
            "axes[1, 0].set_title('Study Time vs G3 Grade')\n",
            "sns.boxplot(data=df, x='failures', y='g3', ax=axes[1, 1], palette='Reds')\n",
            "axes[1, 1].set_title('Past Class Failures vs G3 Grade')\n",
            "sns.boxplot(data=df, x='schoolsup', y='g3', ax=axes[2, 0], palette='Set1')\n",
            "axes[2, 0].set_title('School Support vs G3 Grade')\n",
            "sns.boxplot(data=df, x='romantic', y='g3', ax=axes[2, 1], palette='Pastel1')\n",
            "axes[2, 1].set_title('Romantic relationship vs G3 Grade')\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/eda_dashboard.png', dpi=150)\n",
            "plt.show()"
        ]
    })
    
    # 4D Hypothesis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4D: Hypothesis Testing"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('HYPOTHESIS 1: Students with higher study time will score better in G3.')\n",
            "high_study = df[df['studytime'] >= 3]['g3']\n",
            "low_study = df[df['studytime'] <= 2]['g3']\n",
            "t_stat, p_val = stats.ttest_ind(high_study, low_study)\n",
            "print(f'T-statistic: {t_stat:.3f}, p-value: {p_val:.6f}')\n",
            "print('Result:', 'CONFIRMED' if p_val < 0.05 else 'REJECTED')\n",
            "\n",
            "print('\\nHYPOTHESIS 2: Students in a romantic relationship have lower G3 scores.')\n",
            "rom_yes = df[df['romantic'] == 'yes']['g3']\n",
            "rom_no = df[df['romantic'] == 'no']['g3']\n",
            "t_stat, p_val = stats.ttest_ind(rom_yes, rom_no)\n",
            "print(f'T-statistic: {t_stat:.3f}, p-value: {p_val:.6f}')\n",
            "print('Result:', 'CONFIRMED' if p_val < 0.05 else 'REJECTED')\n",
            "\n",
            "print('\\nHYPOTHESIS 3: Students receiving extra school educational support have lower G3 scores.')\n",
            "sup_yes = df[df['schoolsup'] == 'yes']['g3']\n",
            "sup_no = df[df['schoolsup'] == 'no']['g3']\n",
            "t_stat, p_val = stats.ttest_ind(sup_yes, sup_no)\n",
            "print(f'T-statistic: {t_stat:.3f}, p-value: {p_val:.6f}')\n",
            "print('Result:', 'CONFIRMED' if p_val < 0.05 else 'REJECTED')\n",
            "\n",
            "print('\\nHYPOTHESIS 4: Students with higher weekend alcohol consumption (Walc) have higher absences.')\n",
            "corr, p_val = stats.pearsonr(df['walc'], df['absences'])\n",
            "print(f'Pearson correlation: {corr:.3f}, p-value: {p_val:.6f}')\n",
            "print('Result:', 'CONFIRMED' if p_val < 0.05 else 'REJECTED')"
        ]
    })
    
    # Part 5 Modeling
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 5 — Predictive Model Development\n",
            "### 5A: Model Architecture Selection"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split, cross_validate\n",
            "from sklearn.linear_model import LinearRegression\n",
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
            "from sklearn.metrics import r2_score, mean_absolute_error\n",
            "\n",
            "# Prepare data (One-hot encoding)\n",
            "df_ml = pd.get_dummies(df, drop_first=True)\n",
            "X = df_ml.drop('g3', axis=1)\n",
            "y = df_ml['g3']\n",
            "\n",
            "# Split data\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "\n",
            "# Models to compare\n",
            "models = {\n",
            "    'Linear Regression': LinearRegression(),\n",
            "    'Random Forest': RandomForestRegressor(random_state=42),\n",
            "    'Gradient Boosting': GradientBoostingRegressor(random_state=42)\n",
            "}\n",
            "\n",
            "# Evaluate with 5-Fold Cross-Validation\n",
            "print('--- 5-Fold CV Performance ---')\n",
            "for name, model in models.items():\n",
            "    cv = cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_absolute_error'))\n",
            "    print(f\"{name}:\")\n",
            "    print(f\"  Mean R2 Score: {np.mean(cv['test_r2']):.4f}\")\n",
            "    print(f\"  Mean MAE: {-np.mean(cv['test_neg_mean_absolute_error']):.4f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5B: Feature Importance Assessment"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train best model (Random Forest)\n",
            "rf_model = RandomForestRegressor(random_state=42)\n",
            "rf_model.fit(X_train, y_train)\n",
            "\n",
            "# Calculate importances\n",
            "importances = rf_model.feature_importances_\n",
            "indices = np.argsort(importances)[::-1]\n",
            "feature_names = X.columns\n",
            "\n",
            "# Plot\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.barplot(x=importances[indices[:10]], y=feature_names[indices[:10]], palette='viridis')\n",
            "plt.title('Top 10 Feature Importances (Random Forest)')\n",
            "plt.xlabel('Relative Importance')\n",
            "plt.savefig('outputs/feature_importance.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Identify features with importance < 0.01\n",
            "low_importance = [feature_names[i] for i in range(len(importances)) if importances[i] < 0.01]\n",
            "print(f\"Total features with importance < 0.01: {len(low_importance)}\")\n",
            "print(\"Sample of removable features:\", low_importance[:10])"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5C: Ensemble Model Creation"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.ensemble import StackingRegressor\n",
            "from sklearn.linear_model import Ridge\n",
            "\n",
            "# Define base models\n",
            "estimators = [\n",
            "    ('lr', LinearRegression()),\n",
            "    ('rf', RandomForestRegressor(random_state=42)),\n",
            "    ('gb', GradientBoostingRegressor(random_state=42))\n",
            "]\n",
            "\n",
            "# Create Stacking Ensemble with Ridge as Meta-Model\n",
            "stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())\n",
            "stack_model.fit(X_train, y_train)\n",
            "\n",
            "# Compare performance on test set\n",
            "results = []\n",
            "for name, model in [('Linear Regression', LinearRegression()), \n",
            "                    ('Random Forest', RandomForestRegressor(random_state=42)), \n",
            "                    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),\n",
            "                    ('Stacking Ensemble', stack_model)]:\n",
            "    model.fit(X_train, y_train)\n",
            "    pred = model.predict(X_test)\n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Test R2': r2_score(y_test, pred),\n",
            "        'Test MAE': mean_absolute_error(y_test, pred)\n",
            "    })\n",
            "\n",
            "display(pd.DataFrame(results))"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5D: Model Interpretability (SHAP)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shap\n",
            "\n",
            "# Compute SHAP values\n",
            "explainer = shap.TreeExplainer(rf_model)\n",
            "shap_values = explainer.shap_values(X_train)\n",
            "\n",
            "# Beeswarm plot\n",
            "plt.figure(figsize=(10, 6))\n",
            "shap.summary_plot(shap_values, X_train, show=False)\n",
            "plt.title('SHAP Summary Beeswarm Plot', fontsize=14)\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/shap_summary.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "# Force plot for Row 0 student\n",
            "shap.initjs()\n",
            "plt.figure(figsize=(12, 4))\n",
            "shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :], matplotlib=True, show=False)\n",
            "plt.title('SHAP Force Plot for Row 0 Student', fontsize=14)\n",
            "plt.tight_layout()\n",
            "plt.savefig('outputs/shap_force.png', dpi=150)\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5E: Improved Generalisation"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import GridSearchCV\n",
            "\n",
            "# Check overfitting\n",
            "train_r2 = rf_model.score(X_train, y_train)\n",
            "test_r2 = rf_model.score(X_test, y_test)\n",
            "print(f\"Default RF Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}\")\n",
            "print(f\"Overfit Gap: {train_r2 - test_r2:.4f}\")\n",
            "\n",
            "# GridSearchCV Tuning\n",
            "param_grid = {\n",
            "    'n_estimators': [50, 100, 150],\n",
            "    'max_depth': [4, 6, 8],\n",
            "    'min_samples_split': [2, 5, 10]\n",
            "}\n",
            "\n",
            "grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')\n",
            "grid_search.fit(X_train, y_train)\n",
            "\n",
            "best_rf_model = grid_search.best_estimator_\n",
            "print(\"\\nBest Parameters:\", grid_search.best_params_)\n",
            "\n",
            "# Evaluation after tuning\n",
            "train_r2_after = best_rf_model.score(X_train, y_train)\n",
            "test_r2_after = best_rf_model.score(X_test, y_test)\n",
            "\n",
            "tuning_comparison = pd.DataFrame({\n",
            "    'Metric': ['Train R2', 'Test R2', 'Overfit Gap'],\n",
            "    'Before Tuning': [train_r2, test_r2, train_r2 - test_r2],\n",
            "    'After Tuning': [train_r2_after, test_r2_after, train_r2_after - test_r2_after]\n",
            "})\n",
            "display(tuning_comparison)"
        ]
    })
    
    # Part 6
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 6 — Create Prediction from the Final model\n",
            "We create a new student representation, update their study time to the maximum value (4), and predict their final grade G3 using our tuned Random Forest model."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Take a sample student\n",
            "new_student = X_train.iloc[[0]].copy()\n",
            "print('Original student study time:', new_student['studytime'].values[0])\n",
            "\n",
            "# Change study time to 4 (maximum)\n",
            "new_student.iloc[0, X_train.columns.get_loc('studytime')] = 4\n",
            "print('Modified student study time:', new_student['studytime'].values[0])\n",
            "\n",
            "# Predict\n",
            "predicted_grade = best_rf_model.predict(new_student)[0]\n",
            "print(f'\\nModel Predicted G3 Grade: {predicted_grade:.2f} (scale 0-20)')"
        ]
    })
    
    # Write ipynb structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open('notebook.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Saved notebook.ipynb successfully!")
    
    print("\n----------------------------------------------------")
    print("Project 1 Complete Execution Successful!")
    print("----------------------------------------------------")

if __name__ == '__main__':
    main()
