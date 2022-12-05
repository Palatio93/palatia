#!/usr/bin/env python
# coding: utf-8

# # **Práctica 4: Métricas de distancia (datos estandarizados)**
# 
# Nombre:
# 
# Número de cuenta:
# 
# Email:

# **Objetivo.** Obtener las matrices de distancia (Euclidiana, Chebyshev, Manhattan, Minkowski) a partir de una matriz de datos.
# 
# 
# **Fuente de datos:**
# 
# * ingresos: son ingresos mensuales de 1 o 2 personas, si están casados.
# * gastos_comunes: son gastos mensuales de 1 o 2 personas, si están casados. 
# * pago_coche
# * gastos_otros
# * ahorros
# * vivienda: valor de la vivienda.
# * estado_civil: 0-soltero, 1-casado, 2-divorciado
# * hijos: cantidad de hijos menores (no trabajan).
# * trabajo: 0-sin trabajo, 1-autonomo, 2-asalariado, 3-empresario, 4-autonomos, 5-asalariados, 6-autonomo y asalariado, 7-empresario y autonomo, 8-empresarios o empresario y autónomo 
# * comprar: 0-alquilar, 1-comprar casa a través de crédito hipotecario con tasa fija a 30 años.
# 

# #### **1) Importar las bibliotecas necesarias**
# 

# In[1]:


import pandas as pd                         
# Para la manipulación y análisis de datos
import numpy as np                          
# Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             
# Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    
# Para el cálculo de distancias
from scipy.spatial import distance


# #### **2) Importar los datos**

# In[2]:


Hipoteca = pd.read_csv("Hipoteca.csv")
Hipoteca


# In[3]:


Hipoteca.info()


# **Estandarización de datos**
# 
# En los algoritmos basados en distancias es fundamental escalar o normalizar los datos para que cada una de las variables contribuyan por igual en el análisis.

# In[42]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler  
estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler
# Con MinMaxScaler tenemos valores entre 0 y 1
MEstandarizada = estandarizar.fit_transform(Hipoteca)         # Se calculan la media y desviación y se escalan los datos


# In[43]:


pd.DataFrame(MEstandarizada) 


# #### **3) Matrices de distancia**

# **a) Matriz de distancias: Euclidiana**

# In[44]:


DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
MEuclidiana = pd.DataFrame(DstEuclidiana)


# In[45]:


print(MEuclidiana)
#MEuclidiana 


# In[46]:


print(MEuclidiana.round(3))


# Matriz de distancias de una parte del total de objetos

# In[47]:


DstEuclidiana = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='euclidean')
MEuclidiana = pd.DataFrame(DstEuclidiana)
print(MEuclidiana) 


# Distancia entre dos objetos

# In[48]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
dstEuclidiana 


# **b) Matriz de distancias: Chebyshev**

# In[49]:


DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
MChebyshev = pd.DataFrame(DstChebyshev)


# In[50]:


print(MChebyshev)


# Matriz de distancias de una parte del total de objetos

# In[51]:


DstChebyshev = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='chebyshev')
MChebyshev = pd.DataFrame(DstChebyshev)
print(MChebyshev)


# Distancia entre dos objetos

# In[52]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
dstChebyshev


# **c) Matriz de distancias: Manhattan**

# In[53]:


DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
MManhattan = pd.DataFrame(DstManhattan)


# In[54]:


print(MManhattan)


# Matriz de distancias de una parte del total de objetos

# In[55]:


DstManhattan = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='cityblock')
MManhattan = pd.DataFrame(DstManhattan)
print(MManhattan)


# Distancia entre dos objetos

# In[56]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstManhattan = distance.cityblock(Objeto1,Objeto2)
dstManhattan


# **d) Matriz de distancias: Minkowski**

# In[57]:


DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
MMinkowski = pd.DataFrame(DstMinkowski)


# In[58]:


print(MMinkowski)


# Matriz de distancias de una parte del total de objetos

# In[59]:


DstMinkowski = cdist(MEstandarizada[0:10], MEstandarizada[0:10], metric='minkowski', p=1.5)
MMinkowski = pd.DataFrame(DstMinkowski)
print(MMinkowski)


# Distancia entre dos objetos

# In[60]:


Objeto1 = MEstandarizada[0]
Objeto2 = MEstandarizada[1]
dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=1.5)
dstMinkowski

