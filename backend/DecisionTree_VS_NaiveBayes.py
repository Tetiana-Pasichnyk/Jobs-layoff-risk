import pandas as pd
from config import settings
import config.db_utils
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


conn, _ = config.db_utils.get_db_connection(settings.DB_PORT, settings.DB_PASSWORD)
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

df = pd.read_sql(sql_query, conn)

# ---------- Column definitions ----------
cat_cols_all = ['education_level', 'industry_name', 'job_role_name', 'company_size', 'job_level', 'ai_adoption_level']
num_cols_all = ['age', 'years_of_experience', 'routine_task_percentage', 'creativity_requirement',
                'human_interaction_level', 'number_of_ai_tools_used', 'ai_usage_hours_per_week',
                'tasks_automated_percentage', 'ai_training_hours']

# "significant" = all columns minus the ones we consider weak/noisy predictors
cat_cols_significant = [c for c in cat_cols_all if c != "company_size"]
num_cols_significant = [c for c in num_cols_all if c not in ('age', 'years_of_experience', 'human_interaction_level','tasks_automated_percentage')]

num_cols_without_automated = [c for c in num_cols_all if c !="tasks_automated_percentage"]
num_cols_without_routine = [c for c in num_cols_all if c !="routine_task_percentage"]

y = df['layoff_risk']
class_labels = sorted(y.unique())  # fixed label order so every confusion matrix lines up the same way

# _variabel_name -> es ist a utils for another function, External module calls make no sense
def _build_pipeline(model, cols):
   
    sub_cat = [c for c in cols if c in cat_cols_all]
    sub_num = [c for c in cols if c in num_cols_all]

    """sparse_output=True only records "which row and which column have non-zero values, 
    and what the values are"—zeros are not stored at all, saving memory. 
    sparse_output=False stores the data in its original dense format. 
    DecisionTree works with either format, while GaussianNB requires dense input."""
    if sub_cat:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), sub_cat),
            ('num', 'passthrough', sub_num)
        ])
        return Pipeline([('preprocess', preprocessor), ('model', model)])
    else:
        # purely numerical feature set -> no encoding needed
        return Pipeline([('model', model)])


def run_model(model_factory, cols):
    """
    Train + evaluate one model on one feature set.

    model_factory: zero-arg callable returning a *fresh* estimator,
                   e.g. lambda: GaussianNB()
    cols: list of feature column names to use as input
    """
    pipeline = _build_pipeline(model_factory(), cols)
    x_train, x_test, y_train, y_test = train_test_split(
        df[cols], y, test_size=0.2, random_state=42
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
   
    return pipeline, accuracy


# ---------- The 5 feature sets from the requirements ----------
feature_sets = {
    'automated_only':      ['tasks_automated_percentage'],                 
    'routine_only':["routine_task_percentage"],
    'interaction_only':    ['human_interaction_level'],                   
    'all_features':        num_cols_all + cat_cols_all,                    
    'all_minus_automated':  num_cols_without_automated + cat_cols_all,     
    'all_minus_routine':  num_cols_without_routine + cat_cols_all,   
    'significant_only':    num_cols_significant + cat_cols_significant,   
}

# ---------- The 2 models to compare ----------
model_factories = {
    'DecisionTree': lambda: DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=10),
    'NaiveBayes':   lambda: GaussianNB(),
}

results = {}     # results[set_name][model_name] = accuracy
pipelines = {}   # pipelines[set_name][model_name] = fitted pipeline


for set_name, cols in feature_sets.items():
    results[set_name] = {}
    pipelines[set_name] = {}
    
    for model_name, factory in model_factories.items():
        pipeline, acc = run_model(factory, cols)
        results[set_name][model_name] = acc
        pipelines[set_name][model_name] = pipeline
        
     


# ---------- Comparison table ----------
comparison_df = pd.DataFrame(results).T  # rows = feature sets, cols = models
comparison_df = comparison_df[['DecisionTree', 'NaiveBayes']]

print("\n" + "-" * 60)
print("Accuracy comparison across feature sets and models:")
print("-" * 60)
print(comparison_df.round(4))
print("-" * 60)

best_dt = comparison_df['DecisionTree'].max()
best_dt_ls = comparison_df[comparison_df['DecisionTree'] == best_dt].index.to_list()
best_nb = comparison_df['NaiveBayes'].max()
best_nb_ls = comparison_df[comparison_df['NaiveBayes'] == best_nb].index.to_list()

print(f"\nBest feature set(s) per Decision Tree: {best_dt_ls} (Accuracy: {best_dt:.4f})")
print(f"Best feature set(s) per Naive Bayes:   {best_nb_ls} (Accuracy: {best_nb:.4f})")


# ---------- Predict on new raw data using the 'all_features' models ----------
print("\n" + "-" * 50)
print("Predictions on Test Data:")
print("-" * 50)

# sample 1 (item:11), Medium; sample 2: low; sample 3: high
row_von_database = pd.DataFrame([{
    'education_level': "Master's", 'industry_name': 'Retail', 'job_role_name': 'Inventory Analyst',
    'company_size': 'Small', 'job_level': 'Senior', 'ai_adoption_level': 'Medium',
    'age': 48.0, 'years_of_experience': 6.0, 'routine_task_percentage': 84.0,
    'creativity_requirement': 8.0, 'human_interaction_level': 81.0, 'number_of_ai_tools_used': 2.0,
    'ai_usage_hours_per_week': 5.0, 'tasks_automated_percentage': 67.0, 'ai_training_hours': 29.0
    },
    {
    'education_level': 'PhD', 'industry_name': 'Education', 'job_role_name': 'Research Assistant',
    'company_size': 'Medium', 'job_level': 'Entry', 'ai_adoption_level': 'Low',
    'age': 36.0, 'years_of_experience': 11.0, 'routine_task_percentage': 26.0,
    'creativity_requirement': 60.0, 'human_interaction_level': 77.0, 'number_of_ai_tools_used': 2.0,
    'ai_usage_hours_per_week': 1.0, 'tasks_automated_percentage': 14.0, 'ai_training_hours': 4.0
    },
    {
    'education_level': "Master's", 'industry_name': 'Education', 'job_role_name': 'Teacher',
    'company_size': 'Large', 'job_level': 'Entry', 'ai_adoption_level': 'High',
    'age': 56.0, 'years_of_experience': 11.0, 'routine_task_percentage': 91.0,
    'creativity_requirement': 1.0, 'human_interaction_level': 96.0, 'number_of_ai_tools_used': 9.0,
    'ai_usage_hours_per_week': 26.0, 'tasks_automated_percentage': 74.0, 'ai_training_hours': 68.0
    }
])

# dt-decision tree， nb - naive bayes
model_all_dt = pipelines['all_features']['DecisionTree']
model_all_nb = pipelines['all_features']['NaiveBayes']

print("\nNaive Bayes Predictions:")
for i, pred in enumerate(model_all_nb.predict(row_von_database)):
    print(f"sample {i + 1} Layoff_Risk would be: {pred}")

print("\nDecision Tree Predictions:")
for i, pred in enumerate(model_all_dt.predict(row_von_database)):
    print(f"sample {i + 1} Layoff_Risk would be: {pred}")