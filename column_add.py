import pandas as pd

df1 = pd.read_csv("fingering_plan.csv")
df2 = pd.read_csv("timed_steps.csv")

df_merged = pd.merge(df1, df2[['start_time', 'white_key_index', 'midi']], on='start_time')

df_merged.to_csv('fingering_plan_updated.csv', index=False)