import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["1", "1", "2", "2", "3"],
    "employer" : ["School","Blend360","Blend360","School","School"],
    "date": ["2020-04-13", "2021-11-01", "2021-01-01", "2021-01-11", "2019-03-15"]
})


df["datetime"] = pd.to_datetime(df.date)

count = 0
grouped_df = df.sort_values(by=['name','datetime'], ascending=[True,True])
for i in range(1,len(grouped_df)):
    if grouped_df['employer'][i] == 'Blend360' and grouped_df['employer'][i-1] == 'School':
        count += 1
print(count) 