import pandas as pd
df = pd.read_csv('../input/photos/trainNew.csv')
ds = df.sample(frac=1)

ds.to_csv('../input/photos/shuffled.csv', index=False)
