import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.database import db_utils, settings
from src.ml_model import email_analyzer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

SQL_QUERY = """
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

CAT_COLS = [
    'education_level', 'industry_name', 'job_role_name',
    'company_size', 'job_level', 'ai_adoption_level'
]
NUM_COLS = [
    'age', 'years_of_experience', 'routine_task_percentage', 'creativity_requirement',
    'human_interaction_level', 'number_of_ai_tools_used', 'ai_usage_hours_per_week',
    'tasks_automated_percentage', 'ai_training_hours'
]


def load_data_from_db():
    conn, _ = db_utils.get_db_connection(settings.DB_PORT, settings.DB_PASSWORD)
    return pd.read_sql(SQL_QUERY, conn)


def train_model(df=None):
    if df is None:
        df = load_data_from_db()

    cols = [c for c in df.columns if c not in ('layoff_risk', 'employee_id')]
    sub_cat = [c for c in cols if c in CAT_COLS]
    sub_num = [c for c in cols if c in NUM_COLS]
    y = df['layoff_risk']

    if sub_cat:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), sub_cat),
            ('num', 'passthrough', sub_num)
        ])
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10))
        ])
    else:
        pipeline = Pipeline([
            ('model', DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10))
        ])

    x_train, _, y_train, _ = train_test_split(
        df[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def predict(pipeline, new_rows):
    return pipeline.predict(new_rows)


def main(email_filename="email_sample02.txt"):
    new_data = email_analyzer.huggingface_llama(email_filename)
    pipeline = train_model()
    y_pred = predict(pipeline, new_data)
    print(f"The predicted layoff risk for this position is: {y_pred}")


if __name__ == "__main__":
    main()
