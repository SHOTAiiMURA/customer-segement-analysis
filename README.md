# customer-segement-analysis

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```
```python
df = pd.read_csv('Mall_Customers.csv')
df.head()
```
<img width="521" alt="Screenshot 2024-10-07 at 0 04 37" src="https://github.com/user-attachments/assets/776097ec-b67c-465e-97f0-63f47f4df09d">

```python
df.describe()
```
<img width="547" alt="Screenshot 2024-10-07 at 0 05 26" src="https://github.com/user-attachments/assets/98c1225e-d243-4b00-b8d0-e2bb32be6300">

```python
sns.displot(df["Annual Income (k$)"]);
```
<img width="483" alt="Screenshot 2024-10-07 at 0 05 43" src="https://github.com/user-attachments/assets/34f63600-ded1-4aba-83fc-29fa1168de38">

```python
# name of list using for loop
df.columns
```
<img width="475" alt="Screenshot 2024-10-07 at 0 06 03" src="https://github.com/user-attachments/assets/3aa86871-1a34-4135-91d1-cc2b1157220a">

```python
columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)']
for i in columns:
    sns.displot(df[i]);
```
<img width="494" alt="Screenshot 2024-10-07 at 0 06 51" src="https://github.com/user-attachments/assets/c130fbb2-ed46-43f3-8bfb-71d781b5f424">
<img width="500" alt="Screenshot 2024-10-07 at 0 07 12" src="https://github.com/user-attachments/assets/013bc217-f54d-420f-abbf-3cd113db369c">
<img width="499" alt="Screenshot 2024-10-07 at 0 07 22" src="https://github.com/user-attachments/assets/c85c176e-ec0a-4ad1-9489-7386e938eea0">

```python
sns.kdeplot(df["Annual Income (k$)"], shade=True);
```
<img width="595" alt="Screenshot 2024-10-07 at 0 07 52" src="https://github.com/user-attachments/assets/281e7306-f0bb-493f-892c-7866c07a62b5">

```python
columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    sns.boxplot(data=df,x="Gender",y=df[i]);
```
<img width="590" alt="Screenshot 2024-10-07 at 0 08 31" src="https://github.com/user-attachments/assets/5f78f860-ae18-4c09-b1d1-0fa109b194cd">

```python
df["Gender"].value_counts(normalize=True)
```
<img width="266" alt="Screenshot 2024-10-07 at 0 09 05" src="https://github.com/user-attachments/assets/45626b9f-284a-483e-8c38-895bee4c2594">

```python
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')
```
<img width="600" alt="Screenshot 2024-10-07 at 0 09 25" src="https://github.com/user-attachments/assets/f768d02a-28d6-40b5-a710-79ed45c259e9">

```python
#df=df.drop('CustomerID', axis=1)
sns.pairplot(df, hue="Gender")
```
<img width="748" alt="Screenshot 2024-10-07 at 0 09 43" src="https://github.com/user-attachments/assets/90cece8c-d5b8-47ac-b05f-7eee9351d289">

```python
df.groupby(["Gender"])['Age'].mean()
```
<img width="215" alt="Screenshot 2024-10-07 at 0 10 06" src="https://github.com/user-attachments/assets/ec1a300b-d3df-4bc0-89ce-e006b46a28c0">

```python
df.groupby(["Gender"])['Annual Income (k$)'].mean()
```
<img width="330" alt="Screenshot 2024-10-07 at 0 10 23" src="https://github.com/user-attachments/assets/478636d3-2bb7-40c7-8022-d981ae8c8a5e">

```python
df.groupby(["Gender"])['Spending Score (1-100)'].mean()
```
<img width="357" alt="Screenshot 2024-10-07 at 0 14 49" src="https://github.com/user-attachments/assets/df7d0427-d2bb-4c1f-a4f0-397112251ffa">

```python
# initiate algorithm
clustering1= KMeans(num_cluster= 3)
# call algorithm
clustering1.fit(df[['Annual Income (k$)']])
clustering1.labels_
```
<img width="581" alt="Screenshot 2024-10-07 at 0 15 35" src="https://github.com/user-attachments/assets/c5bc9a04-258b-49dd-9242-06e4b855e1a6">

```python
#8 standard diviation
df['Income Cluster'].value_counts()
```
<img width="205" alt="Screenshot 2024-10-07 at 0 16 04" src="https://github.com/user-attachments/assets/3f454673-77e1-4198-833c-5fb92e722cbb">

```python
intertia_scores = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)
plt.plot(range(1,11),intertia_scores)
# seems elbow stars from 3. now we can use cluster as 3
```
<img width="583" alt="Screenshot 2024-10-07 at 0 16 32" src="https://github.com/user-attachments/assets/43cb207e-0d32-4692-9003-2b5d176f6b3e">

```python
df.groupby("Income Cluster")[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
```
<img width="505" alt="Screenshot 2024-10-07 at 0 17 23" src="https://github.com/user-attachments/assets/b85053ac-bc64-4f20-a097-16f17ea87624">

```python
# Bivariate Clustering - good to understand collelation between "Annual Income" and "Spending Score"
clustering2= KMeans(n_clustering = 5)
#fit data
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
clustering2.labels_
df['Spending and Income Cluster'] =clustering2.labels_
df.head()

intertia_scores2 = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    intertia_scores2.append(kmeans.inertia_)
plt.plot(range(1,11),intertia_scores2)
```
<img width="572" alt="Screenshot 2024-10-07 at 0 18 07" src="https://github.com/user-attachments/assets/2b772df6-fafa-4531-a555-491ea9f30fc2">

```python
centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']
plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df,x='Annual Income (k$)',y='Spending Score (1-100)',hue="Income Cluster", palette="tab10")
```
<img width="739" alt="Screenshot 2024-10-07 at 0 18 34" src="https://github.com/user-attachments/assets/f8cffeb1-4d86-4e22-8ff1-13a8fd175d15">

