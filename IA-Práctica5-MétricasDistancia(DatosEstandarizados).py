#!/usr/bin/env python
# coding: utf-8

# # **Práctica 4: Métricas de distancia (datos estandarizados)**
# 
# Nombre: Humberto Ignacio Hernández Olvera
# 
# Número de cuenta: 309068165
# 
# Email: humberto1nacho@gmail.com

# **Objetivo.** Obtener las matrices de distancia (Euclidiana, Chebyshev, Manhattan, Minkowski) a partir de una matriz de datos.
# 
# 
# **Fuente de datos:**
# 
# Estudios clínicos a partir de imágenes digitalizadas de pacientes con cáncer de mama de Wisconsin (WDBC, Wisconsin Diagnostic Breast Cancer)
# 
# * IDNumber: Identifica al paciente
# * Diagnosis: Diagnóstico (M=Maligno, B=Benigno) 
# * Radius: Media de las distancias del centro y puntos del perímetro
# * Texture: Desviación estándar de la escala de grises
# * Perimeter: Valor del perímetro del cáncer de mama
# * Area: Valor del área del cáncer de mama
# * Smoothness: Variación de la longitud del radio
# * Compactness: Perímetro^2 / Area - 1
# * Concavity: Capida o gravedad de las curvas de nivel
# * Concave points: Número de sectores de contorno cóncavo
# * Symmetry: Simetría de la imagen
# * Fractal dimension: "Aproximación de frontera" - 1
# 

# #### **1) Importar las bibliotecas necesarias**
# 

# In[1]:


import pandas as pd                         # Para la manipulación y análisis de datos
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### **2) Importar los datos**

# In[2]:


Diagnosticos = pd.read_csv("WDBCOriginal.csv")
Diagnosticos


# In[3]:


Diagnosticos.dtypes


# In[4]:


Diagnosticos.isnull().sum()


# Podemos ver que no se tiene ningun valor de tipo nulo.

# In[5]:


plt.plot(Diagnosticos['Texture'],Diagnosticos['Smoothness'],'o')
plt.title('Gráfico de dispersión')
plt.xlabel('Texture')
plt.ylabel('Smoothness')
plt.show()


# In[6]:


sns.scatterplot(x='Texture', y='Smoothness', data=Diagnosticos, hue='Radius')
plt.title('Gráfico de dispersión')
plt.xlabel('Texture')
plt.ylabel('Smoothness')
plt.show()


# In[7]:


plt.plot(Diagnosticos['Perimeter'],Diagnosticos['Area'],'o')
plt.title('Gráfico de dispersión')
plt.xlabel('Perimeter')
plt.ylabel('Area')
plt.show()


# In[8]:


sns.scatterplot(x='Perimeter', y='Area', data=Diagnosticos, hue='Radius')
plt.title('Gráfico de dispersión')
plt.xlabel('Perimeter')
plt.ylabel('Area')
plt.show()


# In[9]:


Diagnosticos.corr()


# In[10]:


plt.figure(figsize=(14,7))
MatrizTnf=np.triu(Diagnosticos.corr())
sns.heatmap(Diagnosticos.corr(),cmap='RdBu_r', annot=True, mask=MatrizTnf)
plt.show()


# Como se observa en el mapa de calor, el perímetro y el área tienen una gran dependencia entre sí debido al radio, y la variable concavity con concavepoints. Se hará la eliminación de las variables perímetro y área con base en el análisis correlacional.

# In[15]:


Diagnosticos = Diagnosticos.drop(columns=['Perimeter','Area']) # Se eliminan las variables con alto grado de similitud


# In[16]:


Diagnosticos = Diagnosticos.drop(columns=['IDNumber','Diagnosis']) # Se eliminan las columnas que no son numéricas


# Habiendo aplicado el análisis correlacional de los datos, se procede a su estandarización.

# **Estandarización de datos**
# 
# En los algoritmos basados en distancias es fundamental escalar o normalizar los datos para que cada una de las variables contribuyan por igual en el análisis.

# In[18]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler  
estandarizar = StandardScaler()   
# Se instancia el objeto StandardScaler o MinMaxScaler 
MEstandarizada = estandarizar.fit_transform(Diagnosticos)         # Se calculan la media y desviación y se escalan los datos


# In[19]:


pd.DataFrame(MEstandarizada) 


# #### **3) Matrices de distancia**

# **a) Matriz de distancias: Euclidiana**

# In[20]:


DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
MEuclidiana = pd.DataFrame(DstEuclidiana)


# In[21]:


print(MEuclidiana)
#MEuclidiana 


# In[22]:


print(MEuclidiana.round(3))


# Matriz de distancias de una parte del total de objetos

# In[23]:


DstEuclidiana = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='euclidean')
MEuclidiana = pd.DataFrame(DstEuclidiana)
print(MEuclidiana) 


# Distancia entre dos objetos

# In[24]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
dstEuclidiana 


# **b) Matriz de distancias: Chebyshev**

# In[25]:


DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
MChebyshev = pd.DataFrame(DstChebyshev)


# In[26]:


print(MChebyshev)


# Matriz de distancias de una parte del total de objetos

# In[27]:


DstChebyshev = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='chebyshev')
MChebyshev = pd.DataFrame(DstChebyshev)
print(MChebyshev)


# Distancia entre dos objetos

# In[28]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
dstChebyshev


# **c) Matriz de distancias: Manhattan**

# In[29]:


DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
MManhattan = pd.DataFrame(DstManhattan)


# In[30]:


print(MManhattan)


# Matriz de distancias de una parte del total de objetos

# In[31]:


DstManhattan = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='cityblock')
MManhattan = pd.DataFrame(DstManhattan)
print(MManhattan)


# Distancia entre dos objetos

# In[32]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstManhattan = distance.cityblock(Objeto1,Objeto2)
dstManhattan


# **d) Matriz de distancias: Minkowski**

# In[33]:


DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
MMinkowski = pd.DataFrame(DstMinkowski)


# In[34]:


print(MMinkowski)


# Matriz de distancias de una parte del total de objetos

# In[35]:


DstMinkowski = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='minkowski', p=1.5)
MMinkowski = pd.DataFrame(DstMinkowski)
print(MMinkowski)


# Distancia entre dos objetos

# In[36]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=1.5)
dstMinkowski

