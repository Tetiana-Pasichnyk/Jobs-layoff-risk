import pandas as pd
import numpy as np
import mysql.connector

print("=== STARTING DATA CLEANING & INGESTION PROCESS ===")

# ==========================================
# 1. DATA LOADING & CLEANING (Pandas)
# ==========================================

# Step 1: Load the raw dataset from the relative path '../data/'
csv_path = '../data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv'
df = pd.read_csv(csv_path)
initial_row_count = len(df)
print(f"Initial dataset loaded from {csv_path}: {initial_row_count} rows.")

# Step 2: Drop rows with missing values in critical baseline columns
critical_columns = ['Layoff_Risk', 'Job_Role', 'Industry']
df_cleaned = df.dropna(subset=critical_columns).copy()
post_drop_count = len(df_cleaned)
dropped_count = initial_row_count - post_drop_count
print(f"[METHOD: DROP] Dropped {dropped_count} rows due to missing critical baseline data.")

# Step 3: Impute missing categorical values using the Mode of the Job_Role group
categorical_columns = ['Education_Level', 'Company_Size', 'Job_Level', 'AI_Adoption_Level']
for col in categorical_columns:
    if df_cleaned[col].isnull().sum() > 0:
        global_mode = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned.groupby('Job_Role')[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else global_mode)
        )

print("[METHOD: IMPUTATION] Textual columns filled using Group Mode.")

# Step 4: Impute missing numerical values using the Median of the Job_Role group
numeric_columns = [
    'Age', 'Years_of_Experience', 'Routine_Task_Percentage', 'Creativity_Requirement',
    'Human_Interaction_Level', 'Number_of_AI_Tools_Used', 'AI_Usage_Hours_Per_Week',
    'Tasks_Automated_Percentage', 'AI_Training_Hours'
]
for col in numeric_columns:
    if df_cleaned[col].isnull().sum() > 0:
        global_median = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned.groupby('Job_Role')[col].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else global_median)
        )

print("[METHOD: IMPUTATION] Numerical columns filled using Group Median.")
print(f"Cleaning Summary: Final Cleaned Rows = {post_drop_count}")
print("--------------------------------------------------")


# ==========================================
# 2. DATABASE CONNECTION & INGESTION (MySQL)
# ==========================================

# Database connection settings (Replace 'your_password' with your actual MySQL password)
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root', 
    'database': 'AI_Impact_DB'
}

try:
    # Step 5: Establish connection to the MySQL server
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    print("Successfully connected to MySQL Server!")
    
    # Step 6: Synchronize unique values with lookup tables using INSERT IGNORE
    print("Syncing lookup tables...")
    
    for edu in df_cleaned['Education_Level'].unique():
        cursor.execute("INSERT IGNORE INTO education_levels (education_level) VALUES (%s)", (edu,))
        
    for ind in df_cleaned['Industry'].unique():
        cursor.execute("INSERT IGNORE INTO industries (industry_name) VALUES (%s)", (ind,))
        
    for role in df_cleaned['Job_Role'].unique():
        cursor.execute("INSERT IGNORE INTO job_roles (job_role_name) VALUES (%s)", (role,))
    
    conn.commit()
    print("Lookup tables synchronized successfully.")
    
    # Step 7: Fetch generated IDs back into Python memory to create mapping dictionaries
    cursor.execute("SELECT education_id, education_level FROM education_levels")
    edu_map = {level: id_ for (id_, level) in cursor.fetchall()}

    cursor.execute("SELECT industry_id, industry_name FROM industries")
    ind_map = {name: id_ for (id_, name) in cursor.fetchall()}

    cursor.execute("SELECT job_role_id, job_role_name FROM job_roles")
    role_map = {name: id_ for (id_, name) in cursor.fetchall()}
    
    # Step 8: Iterate and insert rows into relational tables with explicit type casting
    print("\nWriting data to relational structure (Converting floats to ints where needed)...")
    
    for index, row in df_cleaned.iterrows():
        # 1. Insert into 'employees' table (Casting to INT)
        insert_employee_query = """
            INSERT INTO employees (age, years_of_experience, education_id, industry_id, job_role_id, company_size, job_level)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        employee_data = (
            int(round(row['Age'])),
            int(round(row['Years_of_Experience'])),
            edu_map[row['Education_Level']],
            ind_map[row['Industry']],
            role_map[row['Job_Role']],
            row['Company_Size'],
            row['Job_Level']
        )
        cursor.execute(insert_employee_query, employee_data)
        
        # Get the automatically generated ID of the employee we just inserted
        employee_id = cursor.lastrowid
        
        # 2. Insert into 'ai_impact_metrics' table (Casting to INT, keeping weekly hours as FLOAT)
        insert_metrics_query = """
            INSERT INTO ai_impact_metrics (
                employee_id, routine_task_percentage, creativity_requirement, human_interaction_level,
                ai_adoption_level, number_of_ai_tools_used, ai_usage_hours_per_week,
                tasks_automated_percentage, ai_training_hours, layoff_risk
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        metrics_data = (
            employee_id,
            int(round(row['Routine_Task_Percentage'])),
            int(round(row['Creativity_Requirement'])),
            int(round(row['Human_Interaction_Level'])),
            row['AI_Adoption_Level'],
            int(round(row['Number_of_AI_Tools_Used'])),
            float(row['AI_Usage_Hours_Per_Week']),  # The only field that remains a FLOAT
            int(round(row['Tasks_Automated_Percentage'])),
            int(round(row['AI_Training_Hours'])),
            row['Layoff_Risk']
        )
        cursor.execute(insert_metrics_query, metrics_data)

    # Commit all changes to the disk
    conn.commit()
    print(f"\nSUCCESS! All {len(df_cleaned)} records mapped to INT/FLOAT correctly and saved to MySQL.")

except mysql.connector.Error as err:
    print(f"\n[DATABASE ERROR]: Something went wrong: {err}")

finally:
    # Safely close the database connection
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection closed safely.")