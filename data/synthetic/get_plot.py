import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('synthetic_pareto_train.csv')
upper_quartile=np.percentile(df['U'].values, 50)
plt.hist(df[df.G=='M']['U'].values)
plt.hist(df[df.G=='F']['U'].values)
plt.legend(['Male','Female'])
plt.xlabel('Utility Scores')
plt.ylabel('#Individuals')
plt.savefig('pareto_dist.png',dpi=300)
print(df[df.G=='M'].U.std(),df[df.G=='F'].U.std())
plt.close()

df=pd.read_csv('Rnormal.csv')
plt.hist(df[df.G=='M']['U'])
plt.hist(df[df.G=='F']['U'])
plt.legend(['Male','Female'])
plt.xlabel('Utility Scores')
plt.ylabel('#Individuals')
print(df[df.G=='M'].U.unique(),df[df.G=='F'].U.unique())

plt.savefig('normal_dist.png',dpi=300)

