import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate synthetic "network traffic" data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(300, 2))    # normal
anomaly_data = np.random.uniform(low=-6, high=6, size=(20, 2))   # anomalies

data = np.vstack((normal_data, anomaly_data))
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(scaled_data)

df['Cluster'] = labels
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Cluster', palette='Set2', style='Cluster')
plt.title("DBSCAN – Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Cluster")
plt.show()
