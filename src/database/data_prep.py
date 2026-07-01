import pandas as pd
import numpy as np

def clean_employee_data(df_raw):
    df = df_raw.copy()
    
    all_categorical_cols = [
        'Layoff_Risk', 'Job_Role', 'Industry', 
        'Education_Level', 'Company_Size', 'Job_Level', 'AI_Adoption_Level'
    ]
    for col in all_categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df.replace('nan', np.nan)

    critical_columns = ['Layoff_Risk', 'Job_Role', 'Industry']
    existing_critical = [c for c in critical_columns if c in df.columns]
    df_cleaned = df.dropna(subset=existing_critical).copy()

    categorical_columns = ['Education_Level', 'Company_Size', 'Job_Level', 'AI_Adoption_Level']
    for col in categorical_columns:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            global_mode = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown"
            df_cleaned[col] = df_cleaned.groupby('Job_Role')[col].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else global_mode)
            )

    numeric_columns = [
        'Age', 'Years_of_Experience', 'Routine_Task_Percentage', 'Creativity_Requirement',
        'Human_Interaction_Level', 'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
        'Tasks_Automated_Percentage', 'AI_Training_Hours'
    ]
    for col in numeric_columns:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            global_median = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned.groupby('Job_Role')[col].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median)
            )
            
    return df_cleaned