import pandas as pd
import numpy as np


df=pd.read_csv('fairtrecdata_with_labels_v3.csv')
print(df.groupby('doc_meta').count())

top_cond=df.groupby('doc_meta').count().reset_index().sort_values('doc_id').iloc[-1].doc_meta
df['country']=0
df.loc[df.doc_meta!=top_cond, 'country']=1
df.to_csv('fairtrec_query41_v2.csv',index=False)
