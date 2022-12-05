#!/usr/bin/env python
# coding: utf-8

# ## **Práctica 8: Clustering Jerárquico y Particional**
# 
# Nombre: Humberto Ignacio Hernández Olvera
# 
# No. Cuenta: 309068165
# 
# Email: humberto1nacho@gmail.com

# **Contexto**
# 
# Estudios clínicos a partir de imágenes digitalizadas de pacientes con cáncer de mama de Wisconsin (WDBC, Wisconsin Diagnostic Breast Cancer)
# 
# **Objetivo.** Obtener grupos de pacientes con características similares, diagnosticadas con un tumor de mama, a través de clustering jerárquico y particional.
# 
# **Fuente de datos:**
# 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# - ID number: Identifica al paciente -Discreto-
# - Diagnosis: Diagnostico (M=maligno, B=benigno) -Booleano-
# - Radius: Media de las distancias del centro y puntos del perímetro -Continuo-
# - Texture: Desviación estándar de la escala de grises -Continuo-
# - Perimeter: Valor del perímetro del cáncer de mama -Continuo-
# - Area: Valor del área del cáncer de mama -Continuo-
# - Smoothness: Variación de la longitud del radio -Continuo-
# - Compactness: Perímetro ^ 2 /Area - 1 -Continuo-
# - Concavity: Caída o gravedad de las curvas de nivel -Continuo-
# - Concave points: Número de sectores de contorno cóncavo -Continuo-
# - Symmetry: Simetría de la imagen -Continuo-
# - Fractal dimension: “Aproximación de frontera” - 1 -Continuo-

# #### **1) Importar las bibliotecas necesarias y los datos**

# In[1]:


import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


BCancer = pd.read_csv('WDBCOriginal.csv')
BCancerCheby = BCancer
BCancerCity = BCancer
BCancer


# In[3]:


BCancer.info()


# In[4]:


print(BCancer.groupby('Diagnosis').size())


# #### **2) Selección de características**
# 
#  **Evaluación visual**

# In[5]:


sns.pairplot(BCancer, hue='Diagnosis')
plt.show()


# In[6]:


sns.scatterplot(x='Radius', y ='Perimeter', data=BCancer, hue='Diagnosis')
plt.title('Gráfico de dispersión')
plt.xlabel('Radius')
plt.ylabel('Perimeter')
plt.show()


# In[7]:


sns.scatterplot(x='Concavity', y ='ConcavePoints', data=BCancer, hue='Diagnosis')
plt.title('Gráfico de dispersión')
plt.xlabel('Concavity')
plt.ylabel('ConcavePoints')
plt.show()


# **Matriz de correlaciones**
# 
# Una matriz de correlaciones es útil para analizar la relación entre las variables numéricas.
# Se emplea la función corr()

# In[8]:


CorrBCancer = BCancer.corr(method='pearson')
CorrBCancer 


# In[9]:


print(CorrBCancer['Radius'].sort_values(ascending=False)[:10], '\n')   #Top 10 valores 


# In[10]:


plt.figure(figsize=(14,7))
MatrizInf = np.triu(CorrBCancer)
sns.heatmap(CorrBCancer, cmap='RdBu_r', annot=True, mask=MatrizInf)
plt.show()


# ## Dato importante
# 
# **CARGA DE LA VARIABLE:** Cuanta información te da esa variable.
# De las altamente correlacionadas, quizás no eliminar todas y quedarse solo con una. Probar con distintos escenarios.
# 
# Seleccionar primero las que no tienen correlaciones, que son independientes, con correlaciones moderadas a bajas.

# **Variables seleccionadas:**
# 
# 1) Textura [Posición 3]
# 
# 2) Area [Posición 5]
# 
# 3) Smoothness [Posición 6]
# 
# 4) Compactness [Posición 7]
# 
# 5) Symmetry [Posición 10]
# 
# 6) FractalDimension [Posición 11]

# In[11]:


MatrizVariables = np.array(BCancer[['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']])
pd.DataFrame(MatrizVariables)
#MatrizVariables = BCancer.iloc[:, [3, 5, 6, 7, 10, 11]].values  #iloc para seleccionar filas y columnas


# #### **3) Estandarización de datos**

# In[12]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler  
estandarizar = StandardScaler()                                # Se instancia el objeto StandardScaler o MinMaxScaler 
MEstandarizada = estandarizar.fit_transform(MatrizVariables)   # Se escalan los datos
MEstandarizadaCheby = MEstandarizada
MEstandarizadaCity = MEstandarizada
pd.DataFrame(MEstandarizada) 


# #### **4) Clustering Jerárquico**
# 
# Algoritmo: Ascendente Jerárquico 

# In[13]:


#Se importan las bibliotecas de clustering jerárquico para crear el árbol
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
plt.figure(figsize=(10, 7))
plt.title("Pacientes con cáncer de mama")
plt.xlabel('Observaciones')
plt.ylabel('Distancia')
Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
#plt.axhline(y=7, color='orange', linestyle='--')
#Probar con otras medciones de distancia (euclidean, chebyshev, cityblock)


# In[14]:


#Se crean las etiquetas de los elementos en los clusters
MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean') # affinity=euclidean
MJerarquico.fit_predict(MEstandarizada)
MJerarquico.labels_


# In[15]:


BCancer['clusterH'] = MJerarquico.labels_
BCancer


# In[16]:


#Cantidad de elementos en los clusters
BCancer.groupby(['clusterH'])['clusterH'].count() 


# In[17]:


BCancer[BCancer.clusterH == 0]


# In[18]:


CentroidesH = BCancer.groupby(['clusterH'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
CentroidesH 


# **Clúster 0:** Conformado por 23 pacientes con indicios de cáncer maligno por el tamaño del tumor, con un área promedio de tumor de 775 pixeles y una desviación estándar de textura de 20 pixeles. Aparentemente es un tumor compacto (0.24 pixeles), cuya suavidad alcanza 0.12 pixeles, una simetría de 0.24 y una aproximación de frontera, dimensión fractal, promedio de 0.077 pixeles.
# 
# **Clúster 1:** Es un grupo formado por 88 pacientes, con un tamaño de tumor promedio de 1244 pixeles, lo que nos podría indicar que es un tumor maligno, con compactness de 0.14 pixeles, desviación estándar de textura de 22.5 pixeles, suavidad de 0.09 pixeles, simetría de 0.18 pixeles y dimensión fractal de 0.059 pixeles.
# 
# **Clúster 2:** Es un grupo formado por 248 pacientes, con un tamaño de tumor promedio de 561 pixeles, probablemente benigno, con compactness de 0.11 pixeles, desviación estándar de textura de 18.2 pixeles, suavidad de 0.10 pixeles, simetría de 0.19 pixeles y dimensión fractal de 0.066 pixeles.
# 
# 
# **Clúster 3:** Es un grupo formado por 210 pacientes con el menor tamaño de tumor (posiblemente benigno), con un área promedio de tumor de 505 pixeles y una desviación estándar de textura de 19 pixeles. Es un tumor compacto (0.06 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 

# In[19]:


plt.figure(figsize=(10, 7))
plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
plt.grid()
plt.show()


# #### **5) Clustering particional**
# 
# Algoritmo: k-means

# In[20]:


#Se importan las bibliotecas
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


# In[21]:


#Definición de k clusters para K-means
#Se utiliza random_state para inicializar el generador interno de números aleatorios
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
plt.show()


# **Observación.** 
# En la práctica, puede que no exista un codo afilado (codo agudo) y, como método heurístico, ese "codo" no siempre puede identificarse sin ambigüedades.

# In[22]:


get_ipython().system('pip install kneed')


# In[23]:


from kneed import KneeLocator
kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
kl.elbow


# In[24]:


plt.style.use('ggplot')
kl.plot_knee()


# In[25]:


#Se crean las etiquetas de los elementos en los clusters
MParticional = KMeans(n_clusters=5, random_state=0).fit(MEstandarizada)
MParticional.predict(MEstandarizada)
MParticional.labels_


# In[26]:


BCancer['clusterP'] = MParticional.labels_
BCancer


# In[27]:


BCancer.groupby(['clusterP'])['clusterP'].count()


# In[28]:


BCancer[BCancer.clusterP == 0]


# Obtención de los centroides

# In[29]:


CentroidesP = BCancer.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
CentroidesP


# **Clúster 0:** Es un grupo formado por 85 pacientes con un menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 559 pixeles y una desviación estándar de textura de 24 pixeles. Es un tumor compacto (0.07 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 
# **Clúster 1:** Es un grupo formado por 48 pacientes con un tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 738 pixeles y una desviación estándar de textura de 21 pixeles. Es un tumor no tan compacto (0.21 pixeles), cuya suavidad alcanza 0.12 pixeles, una simetría de 0.23 y una aproximación de frontera, dimensión fractal, promedio de 0.076 pixeles.
# 
# **Clúster 2:** Es un grupo formado por 184 pacientes con un tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 511 pixeles y una desviación estándar de textura de 16 pixeles. Es un tumor compacto (0.06 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 
# **Clúster 3:** Es un grupo formado por 153 pacientes con un menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 480 pixeles y una desviación estándar de textura de 17 pixeles. Es un tumor compacto (0.11 pixeles), cuya suavidad alcanza 0.11 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.067 pixeles.
# 
# **Clúster 4:** Es un grupo formado por 99 pacientes con el mayor tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 1231 pixeles y una desviación estándar de textura de 22 pixeles. Es un tumor compacto (0.14 pixeles), cuya suavidad alcanza 0.09 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.

# In[30]:


# Gráfica de los elementos y los centros de los clusters
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (10, 7)
plt.style.use('ggplot')
colores=['red', 'blue', 'green', 'yellow', 'cyan']
asignar=[]
for row in MParticional.labels_:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(MEstandarizada[:, 0], 
           MEstandarizada[:, 1], 
           MEstandarizada[:, 2], marker='o', c=asignar, s=60)
ax.scatter(MParticional.cluster_centers_[:, 0], 
           MParticional.cluster_centers_[:, 1], 
           MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
plt.show()


# ## Clustering Jeráquico con Chebyshev

# In[31]:


plt.figure(figsize=(10, 7))
plt.title("Pacientes con cáncer de mama")
plt.xlabel('Observaciones')
plt.ylabel('Distancia')
ArbolCheby = shc.dendrogram(shc.linkage(MEstandarizadaCheby, method='complete', metric='chebyshev'))
#plt.axhline(y=7, color='orange', linestyle='--')
#Probar con otras medciones de distancia (euclidean, chebyshev, cityblock)


# In[32]:


MJerarquicoCheby = AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='chebyshev') # affinity=euclidean
MJerarquicoCheby.fit_predict(MEstandarizadaCheby)
MJerarquicoCheby.labels_


# In[33]:


# BCancerCheby = BCancer
BCancerCheby['clusterH'] = MJerarquicoCheby.labels_
BCancerCheby


# In[34]:


#Cantidad de elementos en los clusters
BCancerCheby.groupby(['clusterH'])['clusterH'].count() 


# In[35]:


BCancerCheby[BCancerCheby.clusterH == 0]


# In[36]:


CentroidesHCheby = BCancerCheby.groupby(['clusterH'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
CentroidesHCheby


# **Clúster 0:** Es un grupo formado por 80 pacientes con el menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 569 pixeles y una desviación estándar de textura de 19 pixeles. Es un tumor compacto (0.16 pixeles), cuya suavidad alcanza 0.11 pixeles, una simetría de 0.21 y una aproximación de frontera, dimensión fractal, promedio de 0.073 pixeles.
# 
# **Clúster 1:** Es un grupo formado por 391 pacientes con un tamaño de tumor con un área promedio de 632 pixeles y una desviación estándar de textura de 18 pixeles. Es un tumor muy compacto (0.08 pixeles), cuya suavidad alcanza 0.09 pixeles, una simetría de 0.18 y una aproximación de frontera, dimensión fractal, promedio de 0.061 pixeles.
# 
# **Clúster 2:** Es un grupo formado por 82 pacientes con un tamaño de tumor con un área promedio de 697 pixeles y una desviación estándar de textura de 26 pixeles. Es un tumor compacto (0.09 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.060 pixeles.
# 
# **Clúster 3:** Es un grupo formado por 7 pacientes con el menor tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 2067 pixeles y una desviación estándar de textura de 22 pixeles. Es un tumor compacto (0.21 pixeles), cuya suavidad alcanza 0.11 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.061 pixeles.
# 
# **Clúster 4:** Es un grupo formado por 9 pacientes con el segundo mayor tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 911 pixeles y una desviación estándar de textura de 17 pixeles. Es un tumor compacto (0.26 pixeles), cuya suavidad alcanza 0.13 pixeles, una simetría de 0.26 y una aproximación de frontera, dimensión fractal, promedio de 0.079 pixeles.

# In[37]:


plt.figure(figsize=(10, 7))
plt.scatter(MEstandarizadaCheby[:,0], MEstandarizadaCheby[:,1], c=MJerarquicoCheby.labels_)
plt.grid()
plt.show()


# In[38]:


#Definición de k clusters para K-means
#Se utiliza random_state para inicializar el generador interno de números aleatorios
SSECheby = []
for i in range(2, 12):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(MEstandarizadaCheby)
    SSECheby.append(km.inertia_)

#Se grafica SSE en función de k
plt.figure(figsize=(10, 7))
plt.plot(range(2, 12), SSECheby, marker='o')
plt.xlabel('Cantidad de clusters *k*')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()


# In[39]:


klCheby = KneeLocator(range(2, 12), SSECheby, curve="convex", direction="decreasing")
klCheby.elbow


# In[40]:


plt.style.use('ggplot')
klCheby.plot_knee()


# In[41]:


#Se crean las etiquetas de los elementos en los clusters
MParticionalCheby = KMeans(n_clusters=5, random_state=0).fit(MEstandarizadaCheby)
MParticionalCheby.predict(MEstandarizadaCheby)
MParticionalCheby.labels_


# In[42]:


BCancerCheby['clusterP'] = MParticionalCheby.labels_
BCancerCheby


# In[43]:


BCancerCheby.groupby(['clusterP'])['clusterP'].count()


# In[44]:


BCancerCheby[BCancerCheby.clusterP == 0]


# In[45]:


CentroidesPCheby = BCancerCheby.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
CentroidesPCheby


# **Clúster 0:** Es un grupo formado por 85 pacientes con un menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 559 pixeles y una desviación estándar de textura de 24 pixeles. Es un tumor compacto (0.07 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 
# **Clúster 1:** Es un grupo formado por 48 pacientes con un tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 738 pixeles y una desviación estándar de textura de 21 pixeles. Es un tumor no tan compacto (0.21 pixeles), cuya suavidad alcanza 0.12 pixeles, una simetría de 0.23 y una aproximación de frontera, dimensión fractal, promedio de 0.076 pixeles.
# 
# **Clúster 2:** Es un grupo formado por 184 pacientes con un tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 511 pixeles y una desviación estándar de textura de 16 pixeles. Es un tumor compacto (0.06 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 
# **Clúster 3:** Es un grupo formado por 153 pacientes con un menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 480 pixeles y una desviación estándar de textura de 17 pixeles. Es un tumor compacto (0.11 pixeles), cuya suavidad alcanza 0.11 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.067 pixeles.
# 
# **Clúster 4:** Es un grupo formado por 99 pacientes con el mayor tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 1231 pixeles y una desviación estándar de textura de 22 pixeles. Es un tumor compacto (0.14 pixeles), cuya suavidad alcanza 0.09 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.

# In[46]:


plt.rcParams['figure.figsize'] = (10, 7)
plt.style.use('ggplot')
colores=['red', 'blue', 'green', 'yellow', 'cyan']
asignar=[]
for row in MParticionalCheby.labels_:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(MEstandarizadaCheby[:, 0], 
           MEstandarizadaCheby[:, 1], 
           MEstandarizadaCheby[:, 2], marker='o', c=asignar, s=60)
ax.scatter(MParticionalCheby.cluster_centers_[:, 0], 
           MParticionalCheby.cluster_centers_[:, 1], 
           MParticionalCheby.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
plt.show()


# ## Clustering jerárquico con Cityblock

# In[47]:


plt.figure(figsize=(10, 7))
plt.title("Pacientes con cáncer de mama")
plt.xlabel('Observaciones')
plt.ylabel('Distancia')
ArbolCity = shc.dendrogram(shc.linkage(MEstandarizadaCity, method='complete', metric='cityblock'))
#plt.axhline(y=7, color='orange', linestyle='--')
#Probar con otras medciones de distancia (euclidean, chebyshev, cityblock)


# In[48]:


MJerarquicoCity = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='cityblock') # affinity=euclidean
MJerarquicoCity.fit_predict(MEstandarizadaCity)
MJerarquicoCity.labels_


# In[49]:


# BCancerCity = BCancer
BCancerCity['clusterH'] = MJerarquicoCity.labels_
BCancerCity


# In[50]:


#Cantidad de elementos en los clusters
BCancerCity.groupby(['clusterH'])['clusterH'].count() 


# In[51]:


BCancerCity[BCancerCity.clusterH == 0]


# In[52]:


CentroidesHCity = BCancerCity.groupby(['clusterH'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
CentroidesHCity


# **Clúster 0:** Conformado por 35 pacientes con indicios de cáncer maligno por el tamaño del tumor, con un área promedio de tumor de 775 pixeles y una desviación estándar de textura de 20 pixeles. Aparentemente es un tumor compacto (0.24 pixeles), cuya suavidad alcanza 0.12 pixeles, una simetría de 0.24 y una aproximación de frontera, dimensión fractal, promedio de 0.077 pixeles.
# 
# **Clúster 1:** Es un grupo formado por 88 pacientes, con un tamaño de tumor promedio de 1244 pixeles, lo que nos podría indicar que es un tumor maligno, con compactness de 0.14 pixeles, desviación estándar de textura de 22.5 pixeles, suavidad de 0.09 pixeles, simetría de 0.18 pixeles y dimensión fractal de 0.059 pixeles.
# 
# **Clúster 2:** Es un grupo formado por 248 pacientes, con un tamaño de tumor promedio de 561 pixeles, probablemente benigno, con compactness de 0.11 pixeles, desviación estándar de textura de 18.2 pixeles, suavidad de 0.10 pixeles, simetría de 0.19 pixeles y dimensión fractal de 0.066 pixeles.
# 
# 
# **Clúster 3:** Es un grupo formado por 210 pacientes con el menor tamaño de tumor (posiblemente benigno), con un área promedio de tumor de 505 pixeles y una desviación estándar de textura de 19 pixeles. Es un tumor compacto (0.06 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.

# In[53]:


plt.figure(figsize=(10, 7))
plt.scatter(MEstandarizadaCity[:,0], MEstandarizadaCity[:,1], c=MJerarquicoCity.labels_)
plt.grid()
plt.show()


# In[54]:


#Definición de k clusters para K-means
#Se utiliza random_state para inicializar el generador interno de números aleatorios
SSECity = []
for i in range(2, 12):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(MEstandarizadaCity)
    SSECity.append(km.inertia_)

#Se grafica SSE en función de k
plt.figure(figsize=(10, 7))
plt.plot(range(2, 12), SSECity, marker='o')
plt.xlabel('Cantidad de clusters *k*')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()


# In[55]:


klCity = KneeLocator(range(2, 12), SSECity, curve="convex", direction="decreasing")
klCity.elbow


# In[56]:


plt.style.use('ggplot')
klCity.plot_knee()


# In[57]:


#Se crean las etiquetas de los elementos en los clusters
MParticionalCity = KMeans(n_clusters=5, random_state=0).fit(MEstandarizadaCity)
MParticionalCity.predict(MEstandarizadaCity)
MParticionalCity.labels_


# In[58]:


BCancerCity['clusterP'] = MParticionalCity.labels_
BCancerCity


# In[59]:


BCancerCity.groupby(['clusterP'])['clusterP'].count()


# In[60]:


BCancerCity[BCancerCity.clusterP == 0]


# In[61]:


CentroidesPCity = BCancerCity.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
CentroidesPCity


# **Clúster 0:** Es un grupo formado por 85 pacientes con un menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 559 pixeles y una desviación estándar de textura de 24 pixeles. Es un tumor compacto (0.07 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 
# **Clúster 1:** Es un grupo formado por 48 pacientes con un tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 738 pixeles y una desviación estándar de textura de 21 pixeles. Es un tumor no tan compacto (0.21 pixeles), cuya suavidad alcanza 0.12 pixeles, una simetría de 0.23 y una aproximación de frontera, dimensión fractal, promedio de 0.076 pixeles.
# 
# **Clúster 2:** Es un grupo formado por 184 pacientes con un tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 511 pixeles y una desviación estándar de textura de 16 pixeles. Es un tumor compacto (0.06 pixeles), cuya suavidad alcanza 0.08 pixeles, una simetría de 0.16 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.
# 
# **Clúster 3:** Es un grupo formado por 153 pacientes con un menor tamaño de tumor (potencialmente benigno), con un área promedio de tumor de 480 pixeles y una desviación estándar de textura de 17 pixeles. Es un tumor compacto (0.11 pixeles), cuya suavidad alcanza 0.11 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.067 pixeles.
# 
# **Clúster 4:** Es un grupo formado por 99 pacientes con el mayor tamaño de tumor (potencialmente maligno), con un área promedio de tumor de 1231 pixeles y una desviación estándar de textura de 22 pixeles. Es un tumor compacto (0.14 pixeles), cuya suavidad alcanza 0.09 pixeles, una simetría de 0.19 y una aproximación de frontera, dimensión fractal, promedio de 0.059 pixeles.

# In[62]:


plt.rcParams['figure.figsize'] = (10, 7)
plt.style.use('ggplot')
colores=['red', 'blue', 'green', 'yellow', 'cyan']
asignar=[]
for row in MParticionalCity.labels_:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(MEstandarizadaCity[:, 0], 
           MEstandarizadaCity[:, 1], 
           MEstandarizadaCity[:, 2], marker='o', c=asignar, s=60)
ax.scatter(MParticionalCity.cluster_centers_[:, 0], 
           MParticionalCity.cluster_centers_[:, 1], 
           MParticionalCity.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
plt.show()

