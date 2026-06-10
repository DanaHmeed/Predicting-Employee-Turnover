# -*- coding: utf-8 -*-
"""Employee Turnover Analytics .ipynb
---

## **Machine Learnig**
##**Predicting Employee Turnover Complete Guide Analysis**
---


**Dana Hmeed**

### **The Problem**:
* One of the most common problems at work is turnover.
* Employee turnover is costly, impacts productivity, affects team, and increases recruitment expenses.
*   Replacing a high-level employee can cost multiple of that...

### **Project Objectives**

*  Analyze employee data for patterns related to turnover
* To perform clustering to find any meaningful patterns of employee traits.
* Train ML models to predict future turnover
* Compare model performance
* Recommend retention strategies for high-risk employee groups

"""

import pandas as pd

df = pd.read_csv('HR_comma_sep.csv')
df.head()

#Rename 'sales' column to department
df=df.rename(columns = {'sales':'department'})

"""### **Exploring the Data**"""

#dimesnsions
df.shape

# Check the type of each feature
df.dtypes

df.info()

# Display the statistical overview of the employees
df.describe()

""" ### **Step1: Perform data quality checks by checking for missing values, if any.**"""

#Check for Missing Values and count the sum
df.isnull().sum()

#find out the number of employees who left the company and those who didn’t
df['left'].value_counts()

"""### **Step2 : Heatmap – Correlation between numerical features**

**Summary**:
Heatmap is one of the EDA techniques used to visualize the relationship between variables. It helps identify which features is strongly correlated and have a real impact on the analysis and model building.
* From the heatmap, there is a positive(+) correlation between projectCount, averageMonthlyHours, and evaluation. Which could mean that the employees who spent more hours and did more projects were evaluated highly.

* For the negative(-) relationships, turnover and satisfaction are highly correlated. I'm assuming that people tend to leave a company more when they are less satisfied.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,6))
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of HR Dataset")
plt.show()

"""### **Step2: Distribution Plots (Satisfaction, Evaluation, Hours)**"""

# Employee Satisfaction
plt.figure(figsize=(8,4))
sns.histplot(df['satisfaction_level'], kde=True, color='blue')
plt.title("Distribution of Employee Satisfaction")
plt.show()

# Employee Evaluation
plt.figure(figsize=(8,4))
sns.histplot(df['last_evaluation'], kde=True, color='green')
plt.title("Distribution of Employee Evaluation")
plt.show()

# Employee Monthly Hours
plt.figure(figsize=(8,4))
sns.histplot(df['average_montly_hours'], kde=True, color='orange')
plt.title("Distribution of Average Monthly Hours")
plt.show()

"""### **Step 3: Bar Plot – Number of Projects vs Left**"""

plt.figure(figsize=(8,5))
sns.countplot(x='number_project', hue='left', data=df)
plt.title("Project Count vs Employee Turnover")
plt.show()

"""Inferences from the Bar Plot

* Employees with 2 projects show a high turnover.
This indicates lack of challenge, and low engagement.

* Employees with 6 or 7 projects also show high turnover.
This points to overload, and excessive workload.

* Employees with 3 or 5 projects are the most stable.
These counts represent a healthy workload where most employees stay.

* Overall, both underwork and overwork increase the likelihood of leaving, while a balanced number of projects reduces turnover.
Decision:
Maintain employees at 3–5 concurrent projects to balance productivity and satisfaction.

---

### **Step 3.1: Unsupervised Clustering of Employees Who Left**

Purpose of this step
Before building predictive models, it is important to understand patterns among employees who actually left the company.
Clustering helps HR identify different behavioral groups based on satisfaction and evaluation scores, even without using the target variable.

This is unsupervised learning, meaning:

The algorithm does not use the “left” label for training

It discovers natural patterns inside the data on its own

**Why cluster employees who left?**

Clustering helps answer questions like:

* Are there groups of high performers who still leave?
* Are low-satisfaction employees leaving for similar reasons?
* Do evaluation and satisfaction interact in predictable ways?
"""

# Filter only employees who left the company
left_df = df[df['left'] == 1]
left_df.head()

"""### **Step 3.2: Select the two numerical columns for clustering**"""

X = left_df[['satisfaction_level', 'last_evaluation']]

"""
**Normalize the data K-means works better when features are scaled.**"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**Apply K-means clustering into 3 clusters**

Apply K-Means with 3 clusters

We set k = 3 because:

HR typically works with “high”, “medium”, and “low”-risk groups

The dataset visually supports 3 natural behavioral segments

It keeps interpretation simple and actionable
"""

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

left_df['cluster'] = clusters

"""### **Step3.3: Visualize the clusters**"""

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=left_df['satisfaction_level'],
    y=left_df['last_evaluation'],
    hue=left_df['cluster'],
    palette='viridis'
)
plt.title("K-Means Clustering of Employees Who Left (3 Clusters)")
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.show()

"""### **Step 3.3: Interpretation**

Cluster 0 – Low Satisfaction, Medium Evaluation (purple)

Employees are not satisfied and show average performance.
They appear disengaged and disconnected from their work, which makes them likely to leave due to lack of support or motivation.

Cluster 1 – High Satisfaction, High Evaluation (blue)

Employees are high performers and highly satisfied.
They usually leave for career advancement or better opportunities, not because of unhappiness. They are ambitious and growth-oriented.

Cluster 2 – Very Low Satisfaction, Very High Evaluation (yellow)

Employees show excellent performance but very low satisfaction.
This is the classic burnout group: overworked, undervalued, and stressed.
They are at the highest risk of sudden turnover.

### **Step 4 – Preprocessing + SMOTE for class imbalance.**
**Step 4.1: Separate categorical + numeric columns**
"""

# Separate categorical and numeric
categorical_cols = ['department', 'salary']
numeric_cols = df.drop(columns=categorical_cols).columns

df_numeric = df[numeric_cols]
df_categorical = df[categorical_cols]

"""**Applying get_dummies() to the categorical variables so the model can understand them.**"""

df_categorical_encoded = pd.get_dummies(df_categorical, drop_first=True)

"""**Combine numeric + encoded categorical**"""

df_final = pd.concat([df_numeric, df_categorical_encoded], axis=1)
df_final.head()

"""Define X and y (features + target)"""

X = df_final.drop(columns=['left'])
y = df_final['left']

"""### **Step 4.2: Stratified Train-Test Split (80/20) with random_state=123**

We divide the dataset into two parts:

80% for training the model

20% for testing the model

But we do it using **stratification**, which means:

The proportion of employees who left and employees who stayed is preserved in both the training and testing sets.

This avoids bias.
Without stratification, the test set might accidentally contain mostly “stayed” employees, making evaluation unfair.

**random_state=123 simply ensures you get the same split every time.**
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=123
)

"""### **Step 4.3: Apply SMOTE to Training Set Only**

**SMOTE** creates new, synthetic examples of the minority class instead of just copying rows.
This balances the data so the model can learn fairly from both classes( handle class imbalance).
"""

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=123)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_smote.value_counts())

"""### **Step 5 – Perform 5-fold cross-validation model training and evaluate performance.**

We split the training data into 5 equal parts (folds).

The model is trained 5 times.

Each time:

- 4 folds are used for training

- 1 fold is used for validation

- In the end, we average the results.

Why do we use it?

* To make sure the model is not lucky by chance

* To check that it works well on different data

* To reduce overfitting

Why these algorithms?

* Logistic Regression provides a baseline and interpretable model.

* Random Forest handles nonlinear relationships and prevents overfitting using multiple trees.

* Gradient Boosting builds models sequentially and frequently outperforms RF on structured HR data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""### **5.1 Logistic Regression with 5-Fold Cross Validation**

use Logistic Regression because:

1. The problem is binary classification
0 → Stay
1 → Leave

2. It gives probability outputs HR needs probability scores (like 0.87 = high risk).
"""

# Cross Validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Logistic Regression
log_model = LogisticRegression(max_iter=1500)
y_pred_log = cross_val_predict(log_model, X_train_smote, y_train_smote, cv=cv)

print("Logistic Regression Classification Report:")
print(classification_report(y_train_smote, y_pred_log))

"""### **5.2 Random Forest Classifier with 5-Fold CV**

Random Forest is an ensemble learning method that builds multiple decision trees using random samples and features.
The final prediction is made using majority voting, which improves accuracy and reduces overfitting.
RF detects patterns Logistic Regression cannot.
Captures complex and nonlinear relationships.

"""

rf_model = RandomForestClassifier(n_estimators=200, random_state=123)
y_pred_rf = cross_val_predict(rf_model, X_train_smote, y_train_smote, cv=cv)

print("Random Forest Classification Report:")
print(classification_report(y_train_smote, y_pred_rf))

"""### **5.3 Gradient Boosting Classifier with 5-Fold CV**
Gradient Boosting was used because it builds models sequentially, where each new model learns from the errors of the previous ones. This allows the algorithm to capture complex patterns in employee behavior and results in higher predictive performance for employee turnover.

Gradient Boosting is a powerful ensemble method that builds trees sequentially, where each new tree focuses on fixing the errors made by the previous ones.
"""

gb_model = GradientBoostingClassifier(random_state=123)
y_pred_gb = cross_val_predict(gb_model, X_train_smote, y_train_smote, cv=cv)

print("Gradient Boosting Classification Report:")
print(classification_report(y_train_smote, y_pred_gb))

"""### **Step 6 – Compare Models Using ROC/AUC + Confusion Matrices**

To judge which model performs best in reality we use two main evaluation tools:

ROC Curve & AUC Score

Confusion Matrix
"""

log_model.fit(X_train_smote, y_train_smote)
rf_model.fit(X_train_smote, y_train_smote)
gb_model.fit(X_train_smote, y_train_smote)

"""**ROC Curve**

The ROC curve shows how well a model separates the two classes (Left vs Stayed) at different probability thresholds.

X-axis → False Positive Rate (mistakenly saying “will leave”)

Y-axis → True Positive Rate (correctly identifying those who leave)

**AUC Score (Area Under Curve)**

AUC is a single number that summarizes the entire ROC curve.

* AUC = 0.5 → useless model (same as guessing)

* AUC between 0.7–0.8 → fair

* AUC between 0.8–0.9 → strong

* AUC > 0.9 → excellent

Why we use AUC

Because accuracy becomes unreliable when classes are imbalanced.
AUC ignores class imbalance and focuses on how well the model ranks risky employees.

What this step tells us

Which model is best at distinguishing employees who leave vs stay, regardless of threshol

### **6.1 ROC / AUC and Plot Curves**
"""

models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

plt.figure(figsize=(8,6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} AUC = {auc:.3f}")

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

"""### **6.2 Confusion Matrix for Each Model**

A confusion matrix shows how well the model separates the two classes:

Employees who stayed

Employees who left

It breaks predictions into four simple outcomes:

* TP (True Positive): Model correctly predicts an employee will leave

* FP (False Positive): Model predicts leave but the employee stays

* TN (True Negative): Model correctly predicts an employee will stay

* FN (False Negative): Model predicts stay but the employee actually leaves
"""

def plot_conf(model, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf(log_model, "Logistic Regression Confusion Matrix")
plot_conf(rf_model, "Random Forest Confusion Matrix")
plot_conf(gb_model, "Gradient Boosting Confusion Matrix")

"""### **6.3 Which Metric Matters: Precision or Recall?**

In employee turnover prediction, RECALL is more important than precision.

Because:

If we FAIL to detect an employee who is likely to resign (false negative), the company loses talent.

Missing them means lost productivity, replacement cost, recruitment cost, training cost.

Interpretation to include in report:

Recall is prioritized over precision because the HR department must avoid failing to identify employees who intend to leave. Catching them early allows implementing retention strategies such as salary correction, workload adjustments, or engagement programs.

### **Step 7 – Retention Strategy Based on Predictions**

### **7.1 Predict Probability for Test Data Using Best Model**

Best model: Random Forest

Best AUC score: 0.995

Best confusion matrix: Random Forest (almost perfect classification)

**Best model must be the one that**:

has the highest Recall for class 1

has the highest Precision

has the lowest false negatives (FN)

best confusion matrix

best AUC
"""

# Predict probability of turnover using the best model (Random Forest)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Store results
test_results = X_test.copy()
test_results['Turnover_Probability'] = y_prob

test_results.head()

"""### **Step7.2 Categorize Risk Zones**"""

def categorize(p):
    if p < 0.2:
        return "Safe Zone (Green)"
    elif p < 0.6:
        return "Low Risk (Yellow)"
    elif p < 0.9:
        return "Medium Risk (Orange)"
    else:
        return "High Risk (Red)"

test_results['Risk_Category'] = test_results['Turnover_Probability'].apply(categorize)
test_results.head(10)

"""### **Retention Strategies**

**Safe Zone (Green): Probability < 20 percent**

These employees show very low risk of leaving. They are satisfied, stable, and engaged.**Strategy**: Maintain good conditions, keep engagement steady.

**Low-Risk Zone (Yellow): 20–60 percent**

Employees are mostly stable but show early warning signs (moderate workload pressure).**Strategy**: Light monitoring, regular check-ins, small development opportunities.

**Medium-Risk Zone (Orange): 60–90 percent**

Employees are unhappy, overloaded, or under-recognized They may leave.**Strategy**: Early intervention, adjust workload, offer growth paths, address concerns quickly.

**High-Risk Zone (Red): Probability > 90 percent**

These employees are almost certain to leave.**Strategy**: Immediate retention efforts, address workload/stress, offer incentives or role adjustments.

### **Final Conclusion**

This analysis demonstrates how machine learning can effectively predict employee turnover and support data-driven HR decisions. Using Random Forest as the strongest model (AUC 0.995), accurately identify at-risk employees and categorize them into meaningful risk zones. The clustering analysis highlights distinct behavioral patterns among employees who leave, including burnout, disengagement, and high-performer mobility. Combined with probability-based risk scoring, the organization can implement targeted retention strategies that focus on improving satisfaction, reducing overload, and strengthening career development. By integrating these insights into HR planning, companies can proactively reduce turnover, protect high-performing talent, and build a healthier, more resilient workforce.
"""