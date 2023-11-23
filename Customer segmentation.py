#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Project

# By Praful Kumar

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv("details.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[9]:


df.head(10)


# In[10]:


from sklearn.cluster import KMeans


# In[28]:


df.tail()


# In[29]:


df.tail(10)


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


df.head()


# In[15]:


# Define K-means model
kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)


# In[16]:


# Train the model
kmeans_model.fit(df[['products_purchased','complains',
'money_spent']])


# In[17]:


# To find optimal no of cluster


# In[18]:


# Create the K means model for different values of K
def try_different_clusters(K, data):

    cluster_values = list(range(1, K+1))
    inertias=[]

    for c in cluster_values:
        model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias


# In[19]:


# Find output for k values between 1 to 12 
outputs = try_different_clusters(12, df[['products_purchased','complains','money_spent']])
distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})


# In[21]:


# Finding optimal number of clusters k
figure = go.Figure()
figure.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

figure.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),
                  xaxis_title="Number of clusters",
                  yaxis_title="Sum of squared distances",
                  title_text="Finding optimal number of clusters using elbow method")
figure.show()


# In[22]:


# Re-Train K means model with k=5
kmeans_model_new = KMeans(n_clusters = 5,init='k-means++',max_iter=400,random_state=42)

kmeans_model_new.fit_predict(df[['products_purchased','complains','money_spent']])


# Visualizing customer segments

# In[23]:


# Create data arrays
cluster_centers = kmeans_model_new.cluster_centers_
data = np.expm1(cluster_centers)
points = np.append(data, cluster_centers, axis=1)
points


# In[25]:


# Add "clusters" to customers data
points = np.append(points, [[0], [1], [2], [3], [4]], axis=1)
df["clusters"] = kmeans_model_new.labels_


# In[26]:


df.head()


# In[27]:


# Clusters has been added


# Now we will visualize the clusters using graph

# In[30]:


import plotly.express as px


# In[31]:


import plotly.graph_objects as go


# In[32]:


import matplotlib.pyplot as plt


# In[33]:


figure = px.scatter_3d(df,
                    color='clusters',
                    x="products_purchased",
                    y="complains",
                    z="money_spent",
                    category_orders = {"clusters": ["0", "1", "2", "3", "4"]}
                    )
figure.update_layout()
figure.show()


# In[ ]:


# By pointing our cursor on the cluster we will obtain all the details.

