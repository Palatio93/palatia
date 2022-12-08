import pandas as pd                         
import numpy as np                          
import matplotlib.pyplot as plt             
import seaborn as sns
from scipy.spatial.distance import cdist    
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
# get_ipython().run_line_magic('matplotlib', 'inline')

def get_dataset():
  Diagnosticos = pd.read_csv("WDBCOriginal.csv")
  return Diagnosticos

def print_first():
  Diagnosticos = get_dataset()
  plt.plot(Diagnosticos['Texture'],Diagnosticos['Smoothness'],'o')
  plt.title('Gráfico de dispersión')
  plt.xlabel('Texture')
  plt.ylabel('Smoothness')
  #plt.show()
  plt.savefig('Metricas/metricas1.png')
  plt.savefig('Metricas/metricas1.pdf')

def print_second():
  Diagnosticos = get_dataset()
  sns.scatterplot(x='Texture', y='Smoothness', data=Diagnosticos, hue='Radius')
  plt.title('Gráfico de dispersión')
  plt.xlabel('Texture')
  plt.ylabel('Smoothness')
  #plt.show()
  plt.savefig('Metricas/metricas2.png')
  plt.savefig('Metricas/metricas2.pdf')

def do_corr():
  Diagnosticos = get_dataset()
  Diagnosticos.corr()
  plt.figure(figsize=(14,7))
  MatrizTnf=np.triu(Diagnosticos.corr())
  sns.heatmap(Diagnosticos.corr(),cmap='RdBu_r', annot=True, mask=MatrizTnf)
  #plt.show()
  plt.savefig('Metricas/metricas3.png')
  plt.savefig('Metricas/metricas3.pdf')
  return Diagnosticos

def drop_useless():
  print_first()
  print_second()
  Diagnosticos = do_corr()
  Diagnosticos = Diagnosticos.drop(columns=['Perimeter','Area'])
  Diagnosticos = Diagnosticos.drop(columns=['IDNumber','Diagnosis'])
  return Diagnosticos

def standardize():
  Diagnosticos = drop_useless()
  estandarizar = StandardScaler()
  MEstandarizada = estandarizar.fit_transform(Diagnosticos)

  pd.DataFrame(MEstandarizada)
  return MEstandarizada

def measure():
  MEstandarizada = standardize()
  DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
  MEuclidiana = pd.DataFrame(DstEuclidiana)

  DstEuclidiana = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='euclidean')
  MEuclidiana = pd.DataFrame(DstEuclidiana)
  # print(MEuclidiana)
  return MEuclidiana

def measurepoints(MEstandarizada,pointA=0,pointB=1):
  Objeto1 = MEstandarizada[pointA]
  Objeto2 = MEstandarizada[pointB]
  distanceM = distance.euclidean(Objeto1,Objeto2)
  return distanceM

def main(pointA,pointB):
  MEstandarizada = standardize()
  distanceM = measurepoints(MEstandarizada,pointA,pointB)
  return round(distanceM,3)
