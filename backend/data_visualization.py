import pandas as pd
import numpy as np

file_path = './data/rohdaten-ai-impact-jobs-layoff-risk-dataset.csv'

try:
    df = pd.read_csv(file_path)
    if df is None or df.empty:
        raise ValueError("The data was not retrieved. ")
except Exception as e:
    print(f"failure to extract the file data, errer: {e}")


counts = df.value_counts(['Job_Role','Layoff_Risk'])
print(counts)