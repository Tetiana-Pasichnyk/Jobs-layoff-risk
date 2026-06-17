import pandas as pd
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

def train_model():
    cat_cols = ['Education_Level', 'Industry', 'Job_Role', 'Company_Size', 'Job_Level', 'AI_Adoption_Level']
    num_cols = ['Age', 'Years_of_Experience', 'Routine_Task_Percentage', 'Creativity_Requirement',
                'Human_Interaction_Level', 'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
                'Tasks_Automated_Percentage', 'AI_Training_Hours']
    # drop label column
    cols = [c for c in df_dropna.columns if c != 'Layoff_Risk']
    
    sub_cat = [c for c in cols if c in cat_cols] 
    sub_num = [c for c in cols if c in num_cols]

    y = df_dropna['Layoff_Risk']

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
    return pipeline


def predict(pipeline,new_rows):
    y_pred = pipeline.predict(new_rows)
    return y_pred

new_data = new_row = pd.DataFrame([{
        'Age': 45, 'Education_Level': 'High School', 'Years_of_Experience': 15,
        'Industry': 'Manufacturing', 'Job_Role': 'Operator', 'Company_Size': 'Small',
        'Job_Level': 'Entry', 'Routine_Task_Percentage': 85.0, 'Creativity_Requirement': 10.0,
        'Human_Interaction_Level': 20.0, 'AI_Adoption_Level': 'Low', 'Number_of_AI_Tools_Used': 1.0,
        'AI_Usage_Hours_Per_Week': 2.0, 'Tasks_Automated_Percentage': 70.0, 'AI_Training_Hours': 0.0
    }])

pipeline = train_model()

# -----------  output for Frontend --------------------
y_pred = predict(pipeline,new_data)
print(f"The predicted layoff risk for this position is: {y_pred}")