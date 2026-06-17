import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB         
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

rohdata_path = "./data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv"
try:
    df = pd.read_csv(rohdata_path)
    print("The data has been loaded!")
    if df is None or df.empty:
        raise FileNotFoundError
except Exception as e:
    print(f"something went wrong: {e}")

df_dropna = df.dropna()

cat_cols = ['Education_Level', 'Industry', 'Job_Role', 'Company_Size', 'Job_Level', 'AI_Adoption_Level']
num_cols = ['Age', 'Years_of_Experience', 'Routine_Task_Percentage', 'Creativity_Requirement',
            'Human_Interaction_Level', 'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
            'Tasks_Automated_Percentage', 'AI_Training_Hours']

X = df_dropna.drop(columns=['Layoff_Risk'])
y = df_dropna['Layoff_Risk']


def model_DecisionTree(*variable_col):
    if len(variable_col) == 1 and isinstance(variable_col[0], (tuple, list)):
        cols = list(variable_col[0])
    else:
        cols = list(variable_col)

    # for c in cols, only keep c, wenn c in cat_cols
    sub_cat = [c for c in cols if c in cat_cols] 
    sub_num = [c for c in cols if c in num_cols]

    # handle_unknown='ignore' ensures unseen categories in new data won't raise an error.
    if sub_cat:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), sub_cat),
            # one-hot encode categorical columns, pass the numerical columns through unchanged, and finally concatenate them together
            ('num', 'passthrough', sub_num)
        ])
        # Pipeline: It only fits (learns) the One-Hot encoding rules (e.g., which categories exist) on x_train.
        #  It then applies these learned rules to transform x_train.
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10))
        ])
    else:
        # No categorical columns — skip preprocessing, feed numbers directly to the model.
        pipeline = Pipeline([
            ('model', DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10))
        ])

    x_train, x_test, y_train, y_test = train_test_split(
        df_dropna[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return pipeline, y_test, y_pred

def model_NaiveBayes(*variable_col):              # ← 改函数名
    if len(variable_col) == 1 and isinstance(variable_col[0], (tuple, list)):
        cols = list(variable_col[0])
    else:
        cols = list(variable_col)

    sub_cat = [c for c in cols if c in cat_cols]
    sub_num = [c for c in cols if c in num_cols]

    if sub_cat:
        preprocessor = ColumnTransformer([
            # OHE outputs a Sparse Matrix by default (only stores non-zero positions to save memory).
            # GaussianNB requires Dense, so  set sparse_output=False.
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), sub_cat),
            ('num', 'passthrough', sub_num)
        ])
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', GaussianNB())                # ← 改这里
        ])
    else:
        pipeline = Pipeline([
            ('model', GaussianNB())                # ← 改这里
        ])

    x_train, x_test, y_train, y_test = train_test_split(
        df_dropna[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return pipeline, y_test, y_pred


# --- Decision Tree Model ---
print("\n" + "-" * 50)
print("Decision Tree Model Results:")
print("-" * 50)

model_routine_dt,     y_test_routine_dt,     y_pred_routine_dt     = model_DecisionTree('Routine_Task_Percentage')
model_Interaction_dt, y_test_Interaction_dt, y_pred_Interaction_dt = model_DecisionTree('Human_Interaction_Level')
model_both_dt,        y_test_both_dt,        y_pred_both_dt        = model_DecisionTree('Routine_Task_Percentage', 'Human_Interaction_Level')

# All features (categorical + numerical)
all_columns = num_cols + cat_cols
model_all_dt,  y_test_all_dt,  y_pred_all_dt  = model_DecisionTree(all_columns)

accuracy_routine_dt     = accuracy_score(y_test_routine_dt,     y_pred_routine_dt)
accuracy_Interaction_dt = accuracy_score(y_test_Interaction_dt, y_pred_Interaction_dt)
accuracy_both_dt        = accuracy_score(y_test_both_dt,        y_pred_both_dt)
accuracy_all_dt         = accuracy_score(y_test_all_dt,         y_pred_all_dt)

print(
    f'accuracy_routine:     {accuracy_routine_dt:.4f}\n'
    f'accuracy_Interaction: {accuracy_Interaction_dt:.4f}\n'
    f'accuracy_both:        {accuracy_both_dt:.4f}\n'
    f'accuracy_all:         {accuracy_all_dt:.4f}'
)


# --- Naive Bayes Model ---
print("-" * 50)
print("Naive Bayes Model Results:")
print("-" * 50)

model_routine,     y_test_routine,     y_pred_routine     = model_NaiveBayes('Routine_Task_Percentage')
model_Interaction, y_test_Interaction, y_pred_Interaction = model_NaiveBayes('Human_Interaction_Level')
model_both,        y_test_both,        y_pred_both        = model_NaiveBayes('Routine_Task_Percentage', 'Human_Interaction_Level')

all_columns = num_cols + cat_cols
model_all_nb,  y_test_all_nb,  y_pred_all_nb  = model_NaiveBayes(all_columns)

accuracy_routine     = accuracy_score(y_test_routine,     y_pred_routine)
accuracy_Interaction = accuracy_score(y_test_Interaction, y_pred_Interaction)
accuracy_both        = accuracy_score(y_test_both,        y_pred_both)
accuracy_all_nb      = accuracy_score(y_test_all_nb,      y_pred_all_nb)

print(
    f'accuracy_routine:     {accuracy_routine:.4f}\n'
    f'accuracy_Interaction: {accuracy_Interaction:.4f}\n'
    f'accuracy_both:        {accuracy_both:.4f}\n'
    f'accuracy_all:         {accuracy_all_nb:.4f}'
)


# --- Predict on new raw data using both models ---
print("\n" + "-" * 50)
print("Predictions on New Data:")
print("-" * 50)

new_row = pd.DataFrame([{
        'Age': 45, 'Education_Level': 'High School', 'Years_of_Experience': 15,
        'Industry': 'Manufacturing', 'Job_Role': 'Operator', 'Company_Size': 'Small',
        'Job_Level': 'Entry', 'Routine_Task_Percentage': 85.0, 'Creativity_Requirement': 10.0,
        'Human_Interaction_Level': 20.0, 'AI_Adoption_Level': 'Low', 'Number_of_AI_Tools_Used': 1.0,
        'AI_Usage_Hours_Per_Week': 2.0, 'Tasks_Automated_Percentage': 70.0, 'AI_Training_Hours': 0.0
    },
    {
        'Age': 32, 'Education_Level': 'Master\'s', 'Years_of_Experience': 8,
        'Industry': 'IT', 'Job_Role': 'ML Engineer', 'Company_Size': 'Large',
        'Job_Level': 'Senior', 'Routine_Task_Percentage': 20.0, 'Creativity_Requirement': 90.0,
        'Human_Interaction_Level': 60.0, 'AI_Adoption_Level': 'High', 'Number_of_AI_Tools_Used': 5.0,
        'AI_Usage_Hours_Per_Week': 20.0, 'Tasks_Automated_Percentage': 15.0, 'AI_Training_Hours': 15.0
    },
    {
        'Age': 38, 'Education_Level': 'Bachelor\'s', 'Years_of_Experience': 10,
        'Industry': 'Finance', 'Job_Role': 'Financial Analyst', 'Company_Size': 'Medium',
        'Job_Level': 'Mid', 'Routine_Task_Percentage': 50.0, 'Creativity_Requirement': 50.0,
        'Human_Interaction_Level': 50.0, 'AI_Adoption_Level': 'Medium', 'Number_of_AI_Tools_Used': 3.0,
        'AI_Usage_Hours_Per_Week': 10.0, 'Tasks_Automated_Percentage': 40.0, 'AI_Training_Hours': 5.0
    }])

print("\nNaive Bayes Predictions:")
prediction_nb = model_all_nb.predict(new_row)
for i, pred in enumerate(prediction_nb):
    print(f"sample {i+1} Layoff_Risk would be: {pred}")

print("\nDecision Tree Predictions:")
prediction_dt = model_all_dt.predict(new_row)
for i, pred in enumerate(prediction_dt):
    print(f"sample {i+1} Layoff_Risk would be: {pred}")