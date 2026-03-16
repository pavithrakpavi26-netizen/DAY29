import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import random

# Fix random values
X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_redundant=0,
    n_informative=5,
    n_clusters_per_class=1,
    random_state=42   # important line
)

# Create dataframe
df = pd.DataFrame(X, columns=['col1','col2','col3','col4','col5'])
df['target'] = y

print(df.shape)

print(df.head())


#function for row sampling
def sample_rows(df,percent):
  return df.sample(int(percent*df.shape[0]),replace=True)

#function for features sampling
def sample_features(df,percent):
  cols = random.sample(df.columns.tolist()[:-1],int(percent*(df.shape[1]-1)))
  return df[cols]

#function for combined sampling
def combined_sampling(df,row_percent,col_percent):
  new_df = sample_rows(df,row_percent)
  return sample_features(new_df,col_percent)

df1 = sample_rows(df,0.2) # row sampling
print(df1.shape)
print(df1)

df2 = sample_rows(df,0.2)   # row sampling
print(df2.shape)
print(df2)

df3 = sample_rows(df,0.2) # row sampling
print(df3.shape)
print(df3)

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

clf1.fit(df1.iloc[:,0:5],df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:5],df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:5],df3.iloc[:,-1])

from sklearn.tree import plot_tree

#plot_tree(clf1)
#plot_tree(clf2)
plot_tree(clf3)
