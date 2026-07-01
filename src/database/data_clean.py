import pandas as pd

from src.database.data_prep import clean_employee_data
from src.database.db_utils import get_db_connection
from src.database.settings import DB_PASSWORD, DB_PORT
from src.paths import CSV_PATH

df_csv_raw = pd.read_csv(CSV_PATH)
df_cleaned = clean_employee_data(df_csv_raw)

try:
    conn, cursor = get_db_connection(password=DB_PASSWORD, port=DB_PORT)

    for edu in df_cleaned['Education_Level'].unique():
        cursor.execute("INSERT IGNORE INTO education_levels (education_level) VALUES (%s)", (edu,))

    for ind in df_cleaned['Industry'].unique():
        cursor.execute("INSERT IGNORE INTO industries (industry_name) VALUES (%s)", (ind,))

    for role in df_cleaned['Job_Role'].unique():
        cursor.execute("INSERT IGNORE INTO job_roles (job_role_name) VALUES (%s)", (role,))

    conn.commit()

    cursor.execute("SELECT education_id, education_level FROM education_levels")
    edu_map = {level: id_ for (id_, level) in cursor.fetchall()}

    cursor.execute("SELECT industry_id, industry_name FROM industries")
    ind_map = {name: id_ for (id_, name) in cursor.fetchall()}

    cursor.execute("SELECT job_role_id, job_role_name FROM job_roles")
    role_map = {name: id_ for (id_, name) in cursor.fetchall()}

    inserted_counter = 0
    for _, row in df_cleaned.iterrows():
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
        employee_id = cursor.lastrowid

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
            float(row['AI_Usage_Hours_Per_Week']),
            int(round(row['Tasks_Automated_Percentage'])),
            int(round(row['AI_Training_Hours'])),
            row['Layoff_Risk']
        )
        cursor.execute(insert_metrics_query, metrics_data)
        inserted_counter += 1

    conn.commit()
    print(f"SUCCESS: Inserted {inserted_counter} clean records into MySQL.")

except Exception as err:
    print(f"[ERROR] Database ingestion failed: {err}")
    raise

finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
