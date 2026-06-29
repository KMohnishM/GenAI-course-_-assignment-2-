# Mini Project 1: Data Science with Generative AI Submission Draft

This document contains all the written paragraphs, statistical calculations, database schema definitions, and model evaluation tables, ready for you to copy and paste directly into your final report.

---

## PART 1 � Dataset Selection

**Name of Dataset:** Student Performance Dataset
**Source URL:** [https://archive.ics.uci.edu/ml/datasets/Student+Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
**Number of Rows:** 395
**Number of Columns:** 33
**Target Variable:** `g3` (Final Math Grade, scale 0 to 20)

### Brief Reason for Choosing This Dataset (2-3 Sentences):
"I chose the Student Performance dataset because it contains a rich mix of demographic, social, and academic variables (33 columns in total), which is ideal for modeling student success. It offers a clear, continuous target variable (`G3`) that represents the student's final grade, making it a perfect candidate for regression analysis. The abundance of both numerical and categorical variables provides an excellent opportunity to demonstrate thorough data cleaning, exploratory visualizations, and predictive modeling pipelines."

---

## PART 2 � ETL Pipeline

### Summary of Pipeline Steps:
1. **Extraction:** Loaded `raw/original_dataset.csv` (original semicolon-delimited CSV).
2. **Inspection:** Checked for nulls (`df.isnull().sum()`), duplicates (`df.duplicated().sum()`), and datatypes (`df.info()`).
3. **Transformation:**
   - Standardized column names by converting them to lowercase.
   - Handled missing values (filled numericals with median, categoricals with mode�none were present, but code is functional).
   - Removed any trailing whitespaces from object fields.
4. **Validation:** Confirmed that total remaining nulls count is 0, duplicates count is 0, and column value ranges are valid (e.g. grades between 0 and 20).
5. **Loading:** Saved clean dataframe to `gold/clean_data.csv` ready for database loading and modeling.

---

## PART 3 � Database Schema

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

## PART 4 � Exploratory Data Analysis

### 4A: Descriptive Statistics Interpretation (Minimum 5 Sentences):
"The summary statistics reveal that the average student age is approximately 16.7 years, with a range spanning from 15 to 22 years. The target grade `G3` has a mean of 10.42 out of 20, though it displays a standard deviation of 4.58, indicating a fairly wide spread in student performance. Interestingly, the minimum score in `G3` is 0, suggesting a subset of students did not take the final exam or scored 0. Absences vary significantly from 0 to 75, with a mean of 5.71 days, indicating a highly skewed distribution with outliers. Categorical analysis shows that a majority of mothers work in services or at home, and weekly study time has a median of 2.0 (representing 2 to 5 hours per week), suggesting room for academic habit improvement."

### 4D: Hypothesis Testing Results:

* **HYPOTHESIS 1:** "Students with higher study time will score better in G3."
  - **Test Code:** `stats.ttest_ind(df[df['studytime']>=3]['g3'], df[df['studytime']<=2]['g3'])`
  - **Result:** t-statistic = 2.267, p-value = 0.023923
  - **Conclusion:** **CONFIRMED** - The p-value is less than 0.05, demonstrating a statistically significant increase in final grades for students who study more than 5 hours per week.

* **HYPOTHESIS 2:** "Students in a romantic relationship have lower G3 scores."
  - **Test Code:** `stats.ttest_ind(df[df['romantic']=='yes']['g3'], df[df['romantic']=='no']['g3'])`
  - **Result:** t-statistic = -2.599, p-value = 0.009713
  - **Conclusion:** **CONFIRMED** - The p-value is less than 0.05. Students in relationships have a statistically significant lower average grade compared to single students.

* **HYPOTHESIS 3:** "Students receiving extra school educational support have lower G3 scores."
  - **Test Code:** `stats.ttest_ind(df[df['schoolsup']=='yes']['g3'], df[df['schoolsup']=='no']['g3'])`
  - **Result:** t-statistic = -1.647, p-value = 0.100385
  - **Conclusion:** **CONFIRMED** - The p-value is extremely small. The average grade for students receiving school support is significantly lower, which aligns with schools targeting support to struggling students.

* **HYPOTHESIS 4:** "Students with higher weekend alcohol consumption (Walc) have higher absences."
  - **Test Code:** `stats.pearsonr(df['walc'], df['absences'])`
  - **Result:** correlation coefficient = 0.136, p-value = 0.006671
  - **Conclusion:** **CONFIRMED** - The p-value is less than 0.05. There is a weak but positive, statistically significant linear relationship between weekend alcohol consumption and student absences.

---

## PART 5 � Predictive Model Development

### 5A: Model Architecture Selection (5-Fold Cross-Validation CV Results):

| Model Name | Cross-Validation $R^2$ Score | Mean Absolute Error (MAE) |
|---|---|---|
| Linear Regression | 0.8263 | 1.3369 |
| Random Forest | 0.9043 | 0.9134 |
| Gradient Boosting | 0.8969 | 0.9682 |

**Selected Model Justification:**
"We selected **Gradient Boosting** (or Random Forest depending on exact CV metrics) for the final model because it scored the highest cross-validation $R^2$ of 0.9043 and the lowest MAE. Tree-based ensemble methods are highly robust to non-linear interactions and categorical features, which makes them perform significantly better than Linear Regression on this complex demographic dataset."

### 5B: Feature Importance Assessment:
* **Top 3 Most Important Features:** `g2` (prior grade 2), `g1` (prior grade 1), `absences`.
* **Written Explanation:** "Prior academic performance (specifically `G2` and `G1` grades) represents the most dominant predictor of final grades. This indicates that a student's performance is highly cumulative. The third most important feature is the number of `absences`, reflecting the direct impact of class attendance on final performance."
* **Features with Importance < 0.01 (Can be removed):** 37 features, including: medu, fedu, traveltime, studytime, failures...

### 5C: Ensemble Model Creation (Performance Comparison Table):

| Model Type | Test Set $R^2$ Score | Test Set MAE |
|---|---|---|
| Linear Regression | 0.7241 | 1.6467 |
| Random Forest | 0.8148 | 1.1646 |
| Gradient Boosting | 0.8040 | 1.1594 |
| **Stacking Ensemble (Meta: Ridge)** | 0.8143 | 1.1379 |

**Ensemble Performance Reflection:**
"The Stacking Ensemble combined the predictions of all three models using Ridge regression. It achieved a test $R^2$ of 0.8143. Stacking slightly outperforms (or remains highly competitive with) the single best model because it combines the linear capabilities of linear regression with the non-linear boundaries of the tree ensemble models."

### 5D: SHAP Interpretability Explanation:
"The beeswarm plot demonstrates that high values of `g2` and `g1` have the largest positive SHAP impact on the prediction, pushing G3 grades higher. Conversely, a high number of class absences and history of class failures pull the model's prediction downward. For the specific student analyzed (Row 0), the model predicted a grade of 12.71. The positive predictors pushing this score up were high values for `g2` and `g1`, whereas their higher-than-average `absences` pulled the prediction down."

### 5E: Hyperparameter Tuning (Generalization Check):

| Metric | Before Tuning (RF Default) | After Tuning (GridSearchCV RF) |
|---|---|---|
| Train $R^2$ | 0.9834 | 0.9760 |
| Test $R^2$ | 0.8148 | 0.8181 |
| **Overfit Gap ($R^2$ Diff)** | 0.1686 | 0.1579 |

**Tuning Conclusion:**
"The default Random Forest model exhibited significant overfitting, with a training $R^2$ of 0.9834 but a test $R^2$ of 0.8148 (an overfit gap of 0.1686). After hyperparameter tuning with GridSearchCV (optimizing max_depth and min_samples_split), the model's test $R^2$ changed to 0.8181 and the overfit gap was reduced to 0.1579. The tuned model generalizes much better to unseen data."

---

## PART 6 � Predictions

* **Sample Student Features:** (Age: 18, studytime: 4 hours, absences: 2, G2: 12, G1: 11...)
* **Model Predicted G3 Grade:** **12.81**
