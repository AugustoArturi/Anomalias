#!/usr/bin/env python
# coding: utf-8

# #  Anomaly detection in network traffic

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use("ggplot")
get_ipython().magic(u'matplotlib inline')
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
plotly.tools.set_credentials_file(username='AARTURI', api_key='qiQqOxKJXDlziMFzaB8j')
plotly.offline.init_notebook_mode(connected=True)
from datetime import datetime


# ## Features created from pcap (Wireshark)

# In[ ]:


cols = ["Number","Time","Source","Destination","Protocol","Length","Host","Destination Port","Source Port","Time_delta","Info","Bytes transfer","DateTime", "label"]


# In[ ]:



dfWireshark1 = pd.read_csv("dataset_wireshark" , sep=",", names=cols , index_col=None)
dfWireshark2 = pd.read_csv("dataset_wireshark2", sep=",", names=cols , index_col=None)
dfWireshark3 = pd.read_csv("dataset_wireshark3", sep=",", names=cols , index_col=None)
dfWireshark4 = pd.read_csv("dataset_selfgenerated", sep=",", names=cols , index_col=None)
dfWireshark = pd.concat([dfWireshark1, dfWireshark2, dfWireshark3, dfWireshark4], axis=0, join='inner').reset_index(drop=True)
dfWireshark = dfWireshark.drop("Number", axis = 1)
dfWireshark.to_csv("final_dataset")


# In[ ]:


dfWireshark.head(10)


# ## Preprocessing data

# In[ ]:


dfWireshark.info()


# In[ ]:


#Dense calculation, transforming date timestamps
dfWireshark['DateTime'] = pd.to_datetime(dfWireshark.DateTime,  errors='coerce')


# In[ ]:


dfWireshark.info()


# In[ ]:


#Labeling data in two categories, Normal and Attack traffic
dfWireshark["label"] = "N"
#Attack traffic is hardcoding by previous analysis
dfWireshark.loc[3974:5124,'label'] = "A"
dfWireshark.loc[6745:7063,'label'] = "A"
dfWireshark.loc[8041:8359,'label'] = "A"


# In[ ]:


#verifying behavior
dfWireshark.loc[6745:7063]


# In[ ]:


#verifying behavior
dfWireshark.loc[0:3900]


# In[ ]:


#Rows to delete
dfWireshark.loc[(dfWireshark.label == "A") & (dfWireshark.Protocol != "UDP")]


# In[ ]:


cols_groupby=["label"]
labels = pd.DataFrame({'count':dfWireshark.groupby("label").size()}).reset_index()
labels['count'].sum()


# ### Creating dataframe with value information

# In[ ]:


dfMagicMike = dfWireshark.groupby(['Source',"Protocol","label","Source Port","Destination Port", pd.Grouper(key='DateTime', freq='1s')]).agg({'Destination': pd.Series.nunique}).reset_index()

dfMagicMike.loc[dfMagicMike.Destination > 100]


# In[ ]:


dfMagicMike.head(20)


# In[ ]:


#Correcting wrong labeling
#dfMagicMike = dfMagicMike.loc[((dfMagicMike.label == "A") & (dfMagicMike.Protocol== "UDP") & (dfMagicMike.Destination>1)) | (dfMagicMike.label == "N")]
dfMagicMike.loc[(dfMagicMike.label == "A")]


# In[ ]:


dfTrain = dfMagicMike.copy()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encs = dict()
dfEncoded = dfTrain.copy()  #.sample(frac=1)
for c in dfEncoded.columns:
    if dfEncoded[c].dtype == "object":
        encs[c] = LabelEncoder()
        dfEncoded[c] = encs[c].fit_transform(dfEncoded[c])


# In[ ]:


dfEncoded.groupby("label").describe()


# In[ ]:


#Most important data training
dfEncoded.loc[dfEncoded.label == 0] # 0 = "Attack", 1 = "Normal"


# ## Training data

# In[ ]:


from sklearn.model_selection import train_test_split
dfEncoded2 = dfEncoded.copy()
train_cols = ["Protocol","Destination","Source Port","Destination Port"]
y = dfEncoded.label
X = dfEncoded[train_cols]



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=80)


# ## Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


clf = RandomForestClassifier(max_depth=5, n_estimators = 100,verbose=1, bootstrap=False, random_state=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


dfResults2 = X_test
dfResults2["label"] = y_test
dfResults2["prediction"] = y_pred
dfResults2.loc[(dfResults2.label == 0)]


# # Graphics

# In[ ]:


import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


print(__doc__)

RANDOM_STATE = 12

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(n_estimators=50,
                               warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE))]
   

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 2
max_estimators = 50

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X_train, y_train)


        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()


# In[ ]:


feature = dfMagicMike.label
classnames = dfMagicMike[train_cols]


# Extract single tree
estimator = clf.estimators_[8]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree2.dot', 
                feature_names = classnames.columns.values,
                class_names = feature.to_frame().label.unique(),
                rounded = True, proportion = False, 
                precision = 3, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree2.dot', '-o', 'tree2.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree2.png')


# In[ ]:


clf.estimators_


# In[ ]:




