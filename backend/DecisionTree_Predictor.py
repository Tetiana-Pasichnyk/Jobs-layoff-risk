import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import email_analyzer
from config import db_utils,settings


pd.set_option('display.max_rows', None)      # terminal show all rows
pd.set_option('display.max_columns', None)   # terminal show all cols
pd.set_option('display.width', None)         # width no limit
pd.set_option('display.max_colwidth', None)  # show all content in tabelcell

conn,_ = db_utils.get_db_connection(settings.DB_PORT,settings.DB_PASSWORD)
sql_query = """
        SELECT 
            e.employee_id,
            e.age,
            e.years_of_experience,
            el.education_level,       
            i.industry_name,         
            jr.job_role_name,         
            e.company_size,
            e.job_level,
            m.routine_task_percentage,
            m.creativity_requirement,
            m.human_interaction_level,
            m.ai_adoption_level,
            m.number_of_ai_tools_used,
            m.ai_usage_hours_per_week,
            m.tasks_automated_percentage,
            m.ai_training_hours,
            m.layoff_risk
        FROM employees e
        LEFT JOIN education_levels el ON e.education_id = el.education_id
        LEFT JOIN industries i ON e.industry_id = i.industry_id
        LEFT JOIN job_roles jr ON e.job_role_id = jr.job_role_id
        LEFT JOIN ai_impact_metrics m ON e.employee_id = m.employee_id;
    """

df = pd.read_sql(sql_query,conn)
# print(df.groupby("layoff_risk").size())

def train_model():

    cat_cols_all = ['education_level', 'industry_name', 'job_role_name', 'company_size', 'job_level', 'ai_adoption_level']
    num_cols_all = ['age', 'years_of_experience', 'routine_task_percentage', 'creativity_requirement',
                    'human_interaction_level', 'number_of_ai_tools_used', 'ai_usage_hours_per_week',
                    'tasks_automated_percentage', 'ai_training_hours']

    # "significant" = all columns minus the ones we consider weak/noisy predictors
    cat_cols_significant = [c for c in cat_cols_all if c != "company_size"]
    num_cols_significant = [c for c in num_cols_all if c not in ('age', 'years_of_experience', 'human_interaction_level','tasks_automated_percentage')]

    # drop label column
    cols = [c for c in df.columns if c != 'layoff_risk']
    
    sub_cat = [c for c in cols if c in cat_cols_significant] 
    sub_num = [c for c in cols if c in num_cols_significant]

    y = df['layoff_risk']

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
        df[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def predict(pipeline,new_rows):
    y_pred = pipeline.predict(new_rows)
    return y_pred

# new_data = pd.DataFrame([{
#         'education_level': 'High School',
#         'industry_name': 'Manufacturing', 'job_role_name': 'Operator', 
#         'job_level': 'Entry', 'routine_task_percentage': 85.0, 'creativity_requirement': 10.0,
#         'ai_adoption_level': 'Low', 'number_of_ai_tools_used': 1.0,
#         'ai_usage_hours_per_week': 2.0, 'ai_training_hours': 0.0  
#     }])


new_data = email_analyzer.huggingface_llama("email_sample01.txt")
pipeline = train_model()

# -----------  output for Frontend --------------------
y_pred = predict(pipeline,new_data)

print(f"-"*40," Result ","-"*40)
print(f"The predicted layoff risk for this position is: {y_pred}")
print(f"-"*90)
