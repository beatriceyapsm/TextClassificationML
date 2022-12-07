import pandas as pd
import numpy as np

df=pd.DataFrame({0:["A","B","C"],1:[0.8,0.7,0.9]})
df=df.sort_values(by=1, ascending=False).reset_index(drop = True)
print(df)
A=df.loc[0][1]
print(A)