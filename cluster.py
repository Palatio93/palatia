import pandas as pd                         
import numpy as np                          
import matplotlib.pyplot as plt             
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D

def get_dataset():
  BCancer = pd.read_csv('WDBCOriginal.csv')
  return BCancer

def size_grouped(BCancer):
  return BCancer.groupby('Diagnosis').size()


def first_plot(BCancer):
  sns.pairplot(BCancer, hue='Diagnosis')
  plt.savefig('Cluster/cluster1.png')
  plt.savefig('Cluster/cluster1.pdf')

def second_plot(BCancer):
  sns.scatterplot(x='Radius', y ='Perimeter', data=BCancer, hue='Diagnosis')
  plt.title('Gráfico de dispersión')
  plt.xlabel('Radius')
  plt.ylabel('Perimeter')
  plt.savefig('Cluster/cluster2.png')
  plt.savefig('Cluster/cluster2.pdf')

def do_corr(BCancer):
  CorrBCancer = BCancer.corr(method='pearson')
  return CorrBCancer

def top10_values(CorrBCancer):
  print("Top 10 valores")
  print(CorrBCancer['Radius'].sort_values(ascending=False)[:10], '\n')

def third_plot(CorrBCancer):
  plt.figure(figsize=(14,7))
  MatrizInf = np.triu(CorrBCancer)
  sns.heatmap(CorrBCancer, cmap='RdBu_r', annot=True, mask=MatrizInf)
  plt.savefig('Cluster/cluster3.png')
  plt.savefig('Cluster/cluster3.pdf')

def standardize(BCancer):
  MatrizVariables = np.array(BCancer[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])

  estandarizar = StandardScaler()
  MEstandarizada = estandarizar.fit_transform(MatrizVariables)
  return MEstandarizada

def fourth_plot(MEstandarizada):
  plt.figure(figsize=(10, 7))
  plt.title("Pacientes con cáncer de mama")
  plt.xlabel('Observaciones')
  plt.ylabel('Distancia')
  Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
  plt.savefig('Cluster/cluster4.png')
  plt.savefig('Cluster/cluster4.pdf')
  return Arbol

def hierarching(BCancer, MEstandarizada):
  MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean') # affinity=euclidean
  MJerarquico.fit_predict(MEstandarizada)
  MJerarquico.labels_
  BCancer['clusterH'] = MJerarquico.labels_
  BCancer.groupby(['clusterH'])['clusterH'].count() 
  BCancer[BCancer.clusterH == 0]
  CentroidesH = BCancer.groupby(['clusterH'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
# ** Este es SUPER IMPORTANTE --> CentroidesH
# TODO -> CentroidesH se mandan al usuario 
  plt.figure(figsize=(10, 7))
  plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
  plt.grid()
  plt.savefig('Cluster/cluster5.png')
  plt.savefig('Cluster/cluster5.pdf')

  SSE = []
  for i in range(2, 12):
      km = KMeans(n_clusters=i, random_state=0)
      km.fit(MEstandarizada)
      SSE.append(km.inertia_)

  #Se grafica SSE en función de k
  plt.figure(figsize=(10, 7))
  plt.plot(range(2, 12), SSE, marker='o')
  plt.xlabel('Cantidad de clusters *k*')
  plt.ylabel('SSE')
  plt.title('Elbow Method')
  plt.savefig('Cluster/cluster6.png')
  plt.savefig('Cluster/cluster6.pdf')
  return CentroidesH

def kneeing(BCancer, MEstandarizada):
  SSE = []
  for i in range(2, 12):
      km = KMeans(n_clusters=i, random_state=0)
      km.fit(MEstandarizada)
      SSE.append(km.inertia_)

  kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
  print("Elbow is: ", kl.elbow)

  plt.style.use('ggplot')
  kl.plot_knee()
  MParticional = KMeans(n_clusters=5, random_state=0).fit(MEstandarizada)
  MParticional.predict(MEstandarizada)
  MParticional.labels_

  BCancer['clusterP'] = MParticional.labels_

  # BCancer.groupby(['clusterP'])['clusterP'].count()

  BCancer[BCancer.clusterP == 0]

  CentroidesP = BCancer.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
  return CentroidesP

# TODO: Revisar que tranza con el plot3D
# def plot_3D():
#   plt.rcParams['figure.figsize'] = (10, 7)
#   plt.style.use('ggplot')
#   colores=['red', 'blue', 'green', 'yellow', 'cyan']
#   asignar=[]
#   for row in MParticional.labels_:
#       asignar.append(colores[row])

#   fig = plt.figure()
#   ax = Axes3D(fig)
#   ax.scatter(MEstandarizada[:, 0], 
#             MEstandarizada[:, 1], 
#             MEstandarizada[:, 2], marker='o', c=asignar, s=60)
#   ax.scatter(MParticional.cluster_centers_[:, 0], 
#             MParticional.cluster_centers_[:, 1], 
#             MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)



def main():
  BCancer = get_dataset()
  print("Size of all", size_grouped(BCancer))
  first_plot(BCancer)
  second_plot(BCancer)
  CorrBCancer = do_corr(BCancer)
  print(top10_values(CorrBCancer))
  third_plot(CorrBCancer)
  MEstandarizada = standardize(BCancer)
  fourth_plot(MEstandarizada)
  CentroidesH = hierarching(BCancer,MEstandarizada)
  CentroidesP = kneeing(BCancer, MEstandarizada)
  print(CentroidesP)
  for ind in CentroidesP.index:
    print("Textura: ",CentroidesP["Texture"][ind])
    print("Area: ", CentroidesP["Area"][ind])
  return