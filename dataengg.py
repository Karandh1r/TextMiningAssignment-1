import pandas as pd
import numpy as np


df = pd.DataFrame({
    "name": ["ab1", "ab1", "ab1", "ab1", "ab2", "ab2", "ab3"],
    "date": ["6/1/18", "6/2/18", "6/3/18", "6/4/18", "6/8/18", "6/9/18", "6/23/18"]
})


df["datetime"] = pd.to_datetime(df.date)


grouped_df = df.sort_values(
    ["datetime","name"], ascending=[False,True]
).groupby("name")["date"].apply(list).apply(pd.Series).reset_index()

grouped_df = grouped_df[["name", 0, 1, 2]]
grouped_df.columns = ["name", "most_recent", "second_most_recent","third_most_recent"]
grouped_df["second_most_recent"] = np.where(
    grouped_df.second_most_recent.isna(),
    'NA',
    grouped_df.second_most_recent
)
grouped_df["third_most_recent"] = np.where(
    grouped_df.third_most_recent.isna(),
    'NA',
    grouped_df.third_most_recent
)
print(grouped_df)