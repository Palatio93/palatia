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
  plt.savefig('./static/images/cluster/cluster1.png')
  plt.savefig('./static/images/cluster/cluster1.pdf')

def second_plot(BCancer):
  sns.scatterplot(x='Radius', y ='Perimeter', data=BCancer, hue='Diagnosis')
  plt.title('Gráfico de dispersión')
  plt.xlabel('Radius')
  plt.ylabel('Perimeter')
  plt.savefig('./static/images/cluster/cluster2.png')
  plt.savefig('./static/images/cluster/cluster2.pdf')

def do_corr(BCancer):
  CorrBCancer = BCancer.corr(method='pearson')
  return CorrBCancer

def top10_values(CorrBCancer):
  print("Top 10 valores")
  top10 = CorrBCancer['Radius'].sort_values(ascending=False)[:10]
  print(top10, '\n')
  return top10

def third_plot(CorrBCancer):
  plt.figure(figsize=(14,7))
  MatrizInf = np.triu(CorrBCancer)
  sns.heatmap(CorrBCancer, cmap='RdBu_r', annot=True, mask=MatrizInf)
  plt.savefig('./static/images/cluster/cluster3.png')
  plt.savefig('./static/images/cluster/cluster3.pdf')

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
  plt.savefig('./static/images/cluster/cluster4.png')
  plt.savefig('./static/images/cluster/cluster4.pdf')
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
  plt.savefig('./static/images/cluster/cluster5.png')
  plt.savefig('./static/images/cluster/cluster5.pdf')

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
  plt.savefig('./static/images/cluster/cluster6.png')
  plt.savefig('./static/images/cluster/cluster6.pdf')
  return CentroidesH

def kneeing(BCancer, MEstandarizada):
  SSE = []
  for i in range(2, 12):
      km = KMeans(n_clusters=i, random_state=0)
      km.fit(MEstandarizada)
      SSE.append(km.inertia_)

  kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
  # print("Elbow is: ", kl.elbow)
  elbow = kl.elbow

  plt.style.use('ggplot')
  kl.plot_knee()
  MParticional = KMeans(n_clusters=5, random_state=0).fit(MEstandarizada)
  MParticional.predict(MEstandarizada)
  MParticional.labels_

  BCancer['clusterP'] = MParticional.labels_

  # BCancer.groupby(['clusterP'])['clusterP'].count()

  BCancer[BCancer.clusterP == 0]

  CentroidesP = BCancer.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
  return CentroidesP, elbow

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
  top10 = top10_values(CorrBCancer)
  top10_list = [{
    "Columna1":index,
    "Columna2":round(value,3)
  } for index, value in top10.items()]
  # for index, value in top10.items():
  #   print(type(value))
  #   print(f"Index: {index}, Value: {value}")
  third_plot(CorrBCancer)
  MEstandarizada = standardize(BCancer)
  fourth_plot(MEstandarizada)
  CentroidesH = hierarching(BCancer,MEstandarizada)
  # print(type(CentroidesH))
  CentroidesP, elbow = kneeing(BCancer, MEstandarizada)
  # print(type(CentroidesP))
  # print(CentroidesP)
  CentroidesH_list = [{
    "Cluster": ind,
    "Info": {
      "Textura": round(CentroidesH["Texture"][ind],3),
      "Area": round(CentroidesH["Area"][ind],3),
      "Suavidad": round(CentroidesH["Smoothness"][ind],3),
      "Compacidad": round(CentroidesH["Compactness"][ind],3),
      "Simetria": round(CentroidesH["Symmetry"][ind],3),
      "Dimension_Fractal": round(CentroidesH["FractalDimension"][ind],3)
    }
  } for ind in CentroidesH.index]

  CentroidesP_list = [{
    "Cluster": ind,
    "Info": {
      "Textura": round(CentroidesP["Texture"][ind],3),
      "Area": round(CentroidesP["Area"][ind],3),
      "Suavidad": round(CentroidesP["Smoothness"][ind],3),
      "Compacidad": round(CentroidesP["Compactness"][ind],3),
      "Simetria": round(CentroidesP["Symmetry"][ind],3),
      "Dimension_Fractal": round(CentroidesP["FractalDimension"][ind],3)
    }
  } for ind in CentroidesP.index]
  # for ind in CentroidesP.index:
  #   print("Textura: ",CentroidesP["Texture"][ind])
  #   print("Area: ", CentroidesP["Area"][ind])
  return top10_list, CentroidesH_list, CentroidesP_list, elbow

if __name__ == "__main__":
  main()