import pandas as pd
import numpy as np

rohdata_path = "./data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv"
try:
    df = pd.read_csv(rohdata_path)
    print("The data has been loaded!")
    if df is None or df.empty:
        raise FileNotFoundError
except Exception as e:
    print(f"something went wrong: {e}")

cols = df.columns
# print(f"columns:{cols} ") 
# print(f'len: {len(cols)}')

# x = df['AI_Training_Hours'].unique()
x = df['AI_Training_Hours'].max()
print(x)

# ---------------------------------------- for backup ---------------------------------------------------------------
# cat_cols = ['education_level', 'industry_name', 'job_role_name', 'company_size', 'job_level', 'ai_adoption_level']
# num_cols = ['age', 'years_of_experience', 'routine_task_percentage', 'creativity_requirement','human_interaction_level', 
#             'number_of_ai_tools_used', 'ai_usage_hours_per_week', 'tasks_automated_percentage', 'ai_training_hours']
# -------------------------------------------------------------------------------------------------------------------