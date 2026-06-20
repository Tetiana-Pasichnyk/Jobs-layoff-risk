import pandas as pd
from sklearn.naive_bayes import GaussianNB         
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import config.db_utils
from config import settings


# rohdata_path = "./data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv"
# try:
#     df = pd.read_csv(rohdata_path)
#     print("The data has been loaded!")
#     if df is None or df.empty:
#         raise FileNotFoundError
# except Exception as e:
#     print(f"something went wrong: {e}")

conn,_ = config.db_utils.get_db_connection(settings.DB_PORT,settings.DB_PASSWORD)
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
# df.drop('employee_id')
# colums = df.columns
# print(colums)
# print(f'len:{len(colums)}')


cat_cols = ['education_level', 'industry_name', 'job_role_name', 'company_size', 'job_level', 'ai_adoption_level']
num_cols = ['age', 'years_of_experience', 'routine_task_percentage', 'creativity_requirement','human_interaction_level', 
            'number_of_ai_tools_used', 'ai_usage_hours_per_week', 'tasks_automated_percentage', 'ai_training_hours']
             
                 
y = df['layoff_risk']


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
        df[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return pipeline, y_test, y_pred

def model_NaiveBayes(*variable_col):             
    if len(variable_col) == 1 and isinstance(variable_col[0], (tuple, list)):
        cols = list(variable_col[0])
    else:
        cols = list(variable_col)

    sub_cat = [c for c in cols if c in cat_cols]
    sub_num = [c for c in cols if c in num_cols]

    if sub_cat:
        preprocessor = ColumnTransformer([
            # OHE outputs a Sparse Matrix by default (only stores non-zero positions to save memory).
            # GaussianNB requires Dense, so  set sparse_output=False. ('cat': encoding name, for process repeat )
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), sub_cat),
            ('num', 'passthrough', sub_num)
        ])
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', GaussianNB())  #'model' ModelName             
        ])
    else:
        pipeline = Pipeline([
            ('model', GaussianNB())                
        ])

    x_train, x_test, y_train, y_test = train_test_split(
        df[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return pipeline, y_test, y_pred


# --- Decision Tree Model ---
print("\n" + "-" * 50)
print("Decision Tree Model Results:")
print("-" * 50)

model_routine_dt,     y_test_routine_dt,     y_pred_routine_dt     = model_DecisionTree('routine_task_percentage')
model_Interaction_dt, y_test_Interaction_dt, y_pred_Interaction_dt = model_DecisionTree('human_interaction_level')
model_both_dt,        y_test_both_dt,        y_pred_both_dt        = model_DecisionTree('routine_task_percentage', 'human_interaction_level')

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

model_routine,     y_test_routine,     y_pred_routine     = model_NaiveBayes('routine_task_percentage')
model_Interaction, y_test_Interaction, y_pred_Interaction = model_NaiveBayes('human_interaction_level')
model_both,        y_test_both,        y_pred_both        = model_NaiveBayes('routine_task_percentage', 'human_interaction_level')

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

# sample 1 (item:11)， Medium， sample 2：low， sample 3: high
row_von_database = pd.DataFrame([{
    'education_level': "Master's",
    'industry_name': 'Retail',
    'job_role_name': 'Inventory Analyst',
    'company_size': 'Small',
    'job_level': 'Senior',
    'ai_adoption_level': 'Medium',
    'age': 48.0,
    'years_of_experience': 6.0,
    'routine_task_percentage': 84.0,
    'creativity_requirement': 8.0,
    'human_interaction_level': 81.0,
    'number_of_ai_tools_used': 2.0,
    'ai_usage_hours_per_week': 5.0,
    'tasks_automated_percentage': 67.0,
    'ai_training_hours': 29.0
    },
    {
    'education_level': 'PhD',
    'industry_name': 'Education',
    'job_role_name': 'Research Assistant',
    'company_size': 'Medium',
    'job_level': 'Entry',
    'ai_adoption_level': 'Low',
    'age': 36.0,
    'years_of_experience': 11.0,
    'routine_task_percentage': 26.0,
    'creativity_requirement': 60.0,
    'human_interaction_level': 77.0,
    'number_of_ai_tools_used': 2.0,
    'ai_usage_hours_per_week': 1.0,
    'tasks_automated_percentage': 14.0,
    'ai_training_hours': 4.0
    },
    {
    'education_level': "Master's",
    'industry_name': 'Education',
    'job_role_name': 'Teacher',
    'company_size': 'Large',
    'job_level': 'Entry',
    'ai_adoption_level': 'High',
    'age': 56.0,
    'years_of_experience': 11.0,
    'routine_task_percentage': 91.0,
    'creativity_requirement': 1.0,
    'human_interaction_level': 96.0,
    'number_of_ai_tools_used': 9.0,
    'ai_usage_hours_per_week': 26.0,
    'tasks_automated_percentage': 74.0,
    'ai_training_hours': 68.0
    }
    ])

print("\nNaive Bayes Predictions:")
prediction_nb = model_all_nb.predict(row_von_database)
for i, pred in enumerate(prediction_nb):
    print(f"sample {i+1} Layoff_Risk would be: {pred}")

print("\nDecision Tree Predictions:")
prediction_dt = model_all_dt.predict(row_von_database)
for i, pred in enumerate(prediction_dt):
    print(f"sample {i+1} Layoff_Risk would be: {pred}")