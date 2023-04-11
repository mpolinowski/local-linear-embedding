---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Local Linear Embedding

> [An Introduction to Locally Linear Embedding](https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf): Many problems in information processing involve some form of dimension-
ality reduction. Here we describe locally linear embedding (LLE), an unsu-
pervised learning algorithm that computes low dimensional, neighborhood
preserving embeddings of high dimensional data. LLE attempts to discover
nonlinear structure in high dimensional data by exploiting the local symme-
tries of linear reconstructions. Notably, LLE maps its inputs into a single
global coordinate system of lower dimensionality, and its optimizations—
though capable of generating highly nonlinear embeddings—do not involve
local minima. We illustrate the method on images of lips used in audiovisual
speech synthesis.
> `Lawrence K. Saul`, `Sam T. Roweis`

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding
```

## Dataset

_(see introduction in: [Principal Component Analysis PCA](https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-04-09-principal-component-analysis/2023-04-09))_

```python
raw_data = pd.read_csv('data/A_multivariate_study_of_variation_in_two_species_of_rock_crab_of_genus_Leptograpsus.csv')

data = raw_data.rename(columns={
    'sp' : 'Species',
    'sex' : 'Sex',
    'index' : 'Index',
    'FL' : 'Frontal Lobe',
    'RW' : 'Rear Width',
    'CL' : 'Carapace Midline',
    'CW' : 'Maximum Width',
    'BD' : 'Body Depth'
})

data['Species'] = data['Species'].map({'B':'Blue', 'O':'Orange'})
data['Sex'] = data['Sex'].map({'M':'Male', 'F':'Female'})

data.head(5)
```

|    | Species | Sex | Index | Frontal Lobe | Rear Width | Carapace Midline | Maximum Width | Body Depth |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Blue | Male | 1 | 8.1 | 6.7 | 16.1 | 19.0 | 7.0 |
| 1 | Blue | Male | 2 | 8.8 | 7.7 | 18.1 | 20.8 | 7.4 |
| 2 | Blue | Male | 3 | 9.2 | 7.8 | 19.0 | 22.4 | 7.7 |
| 3 | Blue | Male | 4 | 9.6 | 7.9 | 20.1 | 23.1 | 8.2 |
| 4 | Blue | Male | 5 | 9.8 | 8.0 | 20.3 | 23.0 | 8.2 |

```python
# generate a class variable for all 4 classes
data['Class'] = data.Species + data.Sex

print(data['Class'].value_counts())
data.head(1)
```

* BlueMale: `50`
* BlueFemale: `50`
* OrangeMale: `50`
* OrangeFemale: `50`

|    | species | sex | index | Frontal Lobe | Rear Width | Carapace Midline | Maximum Width | Body Depth | Class |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Blue | Male | 1 | 8.1 | 6.7 | 16.1 | 19.0 | 7.0 | BlueMale |

```python
data_columns = ['Frontal Lobe', 'Rear Width', 'Carapace Midline', 'Maximum Width', 'Body Depth']
```

```python
# normalizing each feature to a given range to make them compareable
data_norm = data.copy()
data_norm[data_columns] = MinMaxScaler().fit_transform(data[data_columns])

data_norm.head()
```

|    | species | sex | index | Frontal Lobe | Rear Width | Carapace Midline | Maximum Width | Body Depth | Class |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Blue | Male | 1 | 0.056604 | 0.014599 | 0.042553 | 0.050667 | 0.058065 | BlueMale |
| 1 | Blue | Male | 2 | 0.100629 | 0.087591 | 0.103343 | 0.098667 | 0.083871 | BlueMale |
| 2 | Blue | Male | 3 | 0.125786 | 0.094891 | 0.130699 | 0.141333 | 0.103226 | BlueMale |
| 3 | Blue | Male | 4 | 0.150943 | 0.102190 | 0.164134 | 0.160000 | 0.135484 | BlueMale |
| 4 | Blue | Male | 5 | 0.163522 | 0.109489 | 0.170213 | 0.157333 | 0.135484 | BlueMale |


## Dimensionality Reduction

The standard [LLE algorithm](https://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding) has the following stages:

* __Nearest Neighbors Search__: The data is projected into a lower dimensional space while trying to preserve distances between neighbors.
* __Weight Matrix Construction__: The weight matrix contains the information that preserves the reconstruction of the input data with fewer dimensions.

```python
# number of components = data columns = 5
# to reduce dimensionality we are going to discard 3
no_components = 3
no_neighbors = 15
lle = LocallyLinearEmbedding(n_components = no_components, n_neighbors = no_neighbors)

data_lle = lle.fit_transform(data_norm[data_columns])

# Note that the reconstruction error increases when adding dimensions
print('Reconstruction Error: ', lle.reconstruction_error_)
# with no_components=3 I get:
# Reconstruction Error:  1.5214133597467682e-05
# with no_components=2:
# Reconstruction Error:  2.1530288023162284e-06

# data_lle contains 1 column for each component
# we can add them to our normalized data set
data_norm[['LLE1', 'LLE2', 'LLE3']] = data_lle

data_norm.head()
```

|    | Species | Sex | Index | Frontal Lobe | Rear Width | Carapace Midline | Maximum Width | Body Depth | Class | LLE1 | LLE2 | LLE3 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Blue | Male | 1 | 0.056604 | 0.014599 | 0.042553 | 0.050667 | 0.058065 | BlueMale | -0.145449 | 0.060973 | 0.092920 |
| 1 | Blue | Male | 2 | 0.100629 | 0.087591 | 0.103343 | 0.098667 | 0.083871 | BlueMale | -0.133111 | 0.057664 | 0.059493 |
| 2 | Blue | Male | 3 | 0.125786 | 0.094891 | 0.130699 | 0.141333 | 0.103226 | BlueMale | -0.126506 | 0.053316 | 0.053484 |
| 3 | Blue | Male | 4 | 0.150943 | 0.102190 | 0.164134 | 0.160000 | 0.135484 | BlueMale | -0.118650 | 0.028331 | 0.059578 |
| 4 | Blue | Male | 5 | 0.163522 | 0.109489 | 0.170213 | 0.157333 | 0.135484 | BlueMale | -0.117088 | 0.022013 | 0.060005 |


### 2D Plot

```python
fig = plt.figure(figsize=(10, 8))
sns.scatterplot(data=data_norm, x='LLE1', y='LLE2', hue='Class')
```

Already the 2d projection allows us to distinguish between the two species - Orange and Blue:

![Local Linear Embedding](https://github.com/mpolinowski/local-linear-embedding/blob/master/assets/Local_Linear_Embedding_01.png)


### 3D Plot

```python
class_colours = {
    'BlueMale': '#0027c4', #blue
    'BlueFemale': '#f18b0a', #orange
    'OrangeMale': '#0af10a', # green
    'OrangeFemale': '#ff1500', #red
}

colours = data['Class'].apply(lambda x: class_colours[x])

x=data_norm.LLE1
y=data_norm.LLE2
z=data_norm.LLE3

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')

ax.scatter(xs=x, ys=y, zs=z, s=50, c=colours)
```

![Local Linear Embedding](https://github.com/mpolinowski/local-linear-embedding/blob/master/assets/Local_Linear_Embedding_02.png)

```python
plot = px.scatter_3d(
    data_norm,
    x = 'LLE1',
    y = 'LLE2',
    z='LLE3',
    color='Class')

plot.show()
```

![Local Linear Embedding](https://github.com/mpolinowski/local-linear-embedding/blob/master/assets/Local_Linear_Embedding_03.png)
