import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import pandas as pd
import numpy as np

from config.settings import MIMIC_PROCESSED_FULL_DATA_PATH

df = pd.read_csv(MIMIC_PROCESSED_FULL_DATA_PATH)
# df_young_adult = pd.read_csv("mimic_processed/mimic_anxiety_dataset_young_adult.csv")

# print(f"DataFrame head:\n {df.head()}")

# print(f"DataFrame All Age shape: {df.shape}")
# print()
# print(f"DataFrame Young Adult shape: {df_young_adult.shape}")

# print("\n")

# print(f"DataFrame All Age info:")
# df.info()
# print()
# print(f"DataFrame Young Adult info:")
# df_young_adult.info()

# print("\n")

# print(f"DataFrame All Age columns: {df.columns.tolist()}")
# print()
# print(f"DataFrame Young Adult columns: {df_young_adult.columns.tolist()}")

# print(f"DataFrame 1st 5 columns in 1st 5 rows:\n {df.iloc[:5, :5]}")
# print(f"DataFrame 2nd 5 columns in 1st 5 rows:\n {df.iloc[:5, 5:10]}")
# print(f"DataFrame 3rd 5 columns in 1st 5 rows:\n {df.iloc[:5, 10:15]}")

# print("\n")

# print(f"Age distribution (All Age):\n {df['age_at_admission'].describe()}")
# print()
# print(f"Age distribution (Young Adult):\n {df_young_adult['age_at_admission'].describe()}")

# print(f"Read clinical notes (All Age):\n {df['clinical_note_text'].iloc[1][:]}")  # print clinical notes one by one with full text
# print()

# for i in range(8):
#     print(f"Read clinical notes (All Age) - Row {i}:\n\n {df['clinical_note_text'].iloc[i][:]}")  # print clinical notes one by one with full text
#     print()

# check null values in the dataset
print(f"Null values in the dataset (All Age):\n {df.isnull().sum()}")

# describe the dastaset
print(f"Dataset description (All Age):\n {df.describe(include='all')}")

# info of the dataset
print(f"Dataset info (All Age):\n {df.info()}")
