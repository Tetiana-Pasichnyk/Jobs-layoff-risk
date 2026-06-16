import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


rohdata_path = "./data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv"
try:
    df = pd.read_csv(rohdata_path)
    print("The data has been loaded!")
    if df is None or df.empty:
        raise FileNotFoundError
except Exception as e:
    print("something went wrong: {e}") 

# no_repeat_content =  df.groupby('Routine_Task_Percentage').size()
# print(type(no_repeat_content))
# print(no_repeat_content.index.tolist())
 
#  Always drop null values on the combined dataframe before splitting into features and labels, 
# otherwise different rows may be dropped separately, causing misalignment and inconsistent sample sizes.
df_dropna = df[['Routine_Task_Percentage','Layoff_Risk']].dropna()


# check the relavent bewteen  Routine_Task_Percentage and Lable, x muss numpy, so train data has to be [[...]]，label: Series,numpy array (1D) or list
x_routine_task = df_dropna[['Routine_Task_Percentage']]
y = df_dropna['Layoff_Risk']
x_train, x_test, y_train, y_test = train_test_split(x_routine_task,y,test_size=0.2, random_state=42)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state= 42,
    min_samples_leaf= 10
)

# 1/(|x|)  (M(X_1 |x_1 | + .…+M(X_k |x_k |)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(y_pred)
print(type(y_pred)) #<class 'numpy.ndarray'>

