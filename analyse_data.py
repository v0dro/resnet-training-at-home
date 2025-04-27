import pandas as pd

train = pd.read_csv("train.out")
test = pd.read_csv("test.out")

print(f"Total training time: {train['iter_time'].sum() / 1e3 / 60 / 60} hours")
print(f"Avg. test accuracy: {test['accuracy'].mean() * 100:.2f}%")