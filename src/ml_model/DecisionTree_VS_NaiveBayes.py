import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.database import db_utils, settings

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


def model_DecisionTree(df, y, *variable_col):
    if len(variable_col) == 1 and isinstance(variable_col[0], (tuple, list)):
        cols = list(variable_col[0])
    else:
        cols = list(variable_col)

    sub_cat = [c for c in cols if c in CAT_COLS]
    sub_num = [c for c in cols if c in NUM_COLS]

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

    x_train, x_test, y_train, y_test = train_test_split(
        df[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return pipeline, y_test, y_pred


def model_NaiveBayes(df, y, *variable_col):
    if len(variable_col) == 1 and isinstance(variable_col[0], (tuple, list)):
        cols = list(variable_col[0])
    else:
        cols = list(variable_col)

    sub_cat = [c for c in cols if c in CAT_COLS]
    sub_num = [c for c in cols if c in NUM_COLS]

    if sub_cat:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), sub_cat),
            ('num', 'passthrough', sub_num)
        ])
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', GaussianNB())
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


def main():
    df = load_data_from_db()
    y = df['layoff_risk']
    all_columns = NUM_COLS + CAT_COLS

    print("\n" + "-" * 50)
    print("Decision Tree Model Results:")
    print("-" * 50)

    _, y_test_routine_dt, y_pred_routine_dt = model_DecisionTree(df, y, 'routine_task_percentage')
    _, y_test_interaction_dt, y_pred_interaction_dt = model_DecisionTree(df, y, 'human_interaction_level')
    _, y_test_both_dt, y_pred_both_dt = model_DecisionTree(
        df, y, 'routine_task_percentage', 'human_interaction_level'
    )
    model_all_dt, y_test_all_dt, y_pred_all_dt = model_DecisionTree(df, y, all_columns)

    print(
        f'accuracy_routine:     {accuracy_score(y_test_routine_dt, y_pred_routine_dt):.4f}\n'
        f'accuracy_Interaction: {accuracy_score(y_test_interaction_dt, y_pred_interaction_dt):.4f}\n'
        f'accuracy_both:        {accuracy_score(y_test_both_dt, y_pred_both_dt):.4f}\n'
        f'accuracy_all:         {accuracy_score(y_test_all_dt, y_pred_all_dt):.4f}'
    )

    print("-" * 50)
    print("Naive Bayes Model Results:")
    print("-" * 50)

    _, y_test_routine, y_pred_routine = model_NaiveBayes(df, y, 'routine_task_percentage')
    _, y_test_interaction, y_pred_interaction = model_NaiveBayes(df, y, 'human_interaction_level')
    _, y_test_both, y_pred_both = model_NaiveBayes(
        df, y, 'routine_task_percentage', 'human_interaction_level'
    )
    model_all_nb, _, _ = model_NaiveBayes(df, y, all_columns)

    print(
        f'accuracy_routine:     {accuracy_score(y_test_routine, y_pred_routine):.4f}\n'
        f'accuracy_Interaction: {accuracy_score(y_test_interaction, y_pred_interaction):.4f}\n'
        f'accuracy_both:        {accuracy_score(y_test_both, y_pred_both):.4f}\n'
        f'accuracy_all:         {accuracy_score(y_test_all_dt, y_pred_all_dt):.4f}'
    )

    print("\n" + "-" * 50)
    print("Predictions on Test Data:")
    print("-" * 50)

    row_von_database = pd.DataFrame([
        {
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
    for i, pred in enumerate(model_all_nb.predict(row_von_database)):
        print(f"sample {i + 1} Layoff_Risk would be: {pred}")

    print("\nDecision Tree Predictions:")
    for i, pred in enumerate(model_all_dt.predict(row_von_database)):
        print(f"sample {i + 1} Layoff_Risk would be: {pred}")


if __name__ == "__main__":
    main()
