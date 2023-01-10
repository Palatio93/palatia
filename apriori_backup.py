#!/usr/bin/env python
# coding: utf-8

# ## **Práctica 3: Reglas de asociación**
# 
# Nombre: Hernández Olvera Humberto Ignacio
# 
# Email: humberto1nacho@gmail.com

# **Objetivo**
# 
# Analizar las transacciones y obtener reglas significativas (patrones) de los programas vistos en una plataforma de streaming.
# No se menciona el periodo de los datos de las transacciones, sin embargo, tomando como referencia la práctica 1 que hicimos, consideraré que tienen un periodo de una semana (7 días) para fines prácticos.
# 
# **Características:**
# 
# * Ítems (32 películas)
# * 9690 transacciones
# 
# 
# Fuente de datos: tv_shows_dataset.csv

# #### **1) Importar las bibliotecas necesarias**

# In[1]:


# ----------- AQUI MODIFICAR --------------

# get_ipython().system('pip install apyori')


# In[2]:


import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori


# #### **2) Importar los datos**

# In[3]:


DatosTransacciones = pd.read_csv('tv_shows_dataset.csv')
DatosTransacciones


# **Observaciones:**
# 
# 1) Se observa que el encabezado es la primera transacción.
# 
# 2) NaN indica que esa película no fue rentada o comprada en esa transacción.

# In[4]:


DatosTransacciones = pd.read_csv('tv_shows_dataset.csv', header=None)
DatosTransacciones


# #### **3) Procesamiento de los datos**
# 
# **Exploración**
# 
# Antes de ejecutar el algoritmo es recomendable observar la distribución de la frecuencia de los elementos.

# In[5]:


#Se incluyen todas las transacciones en una sola lista
Transacciones = DatosTransacciones.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'

#Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
Lista = pd.DataFrame(Transacciones)
Lista['Frecuencia'] = 1

#Se agrupa los elementos
Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
Lista = Lista.rename(columns={0 : 'Item'})

#Se muestra la lista
Lista


# In[6]:


# Se genera un gráfico de barras
plt.figure(figsize=(16,20), dpi=300)
plt.ylabel('Item')
plt.xlabel('Frecuencia')
plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
plt.show()
plt.savefig('fig1.png')
plt.savefig('fig1.pdf')


# In[7]:


#Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
#level=0 especifica desde el primer índice
TransaccionesLista = DatosTransacciones.stack().groupby(level=0).apply(list).tolist()
TransaccionesLista 


# #### **4) Aplicación del algoritmo**

# **Configuración 1**
# 
# Vistas de cada show por día, considerando que es semanal. Buscaré aquellos shows que se hayan visto 20 veces al día (140 a la semana)
# 
# i) El soporte mínimo se calcula de 140/9690 = 0.0144
# 
# ii) La confianza mínima para las reglas será de 25%
# 
# iii) La elevación de 2.

# In[8]:


sup = 140/9690 # .01
confi = 0.25
lifto = 2
ReglasC1 = apriori(TransaccionesLista, 
                   min_support=sup, 
                   min_confidence=confi, 
                   min_lift=lifto)


# Se convierte las reglas encontradas por la clase apriori en una lista, puesto que es más fácil ver los resultados.

# In[9]:


ResultadosC1 = list(ReglasC1)
print(len(ResultadosC1)) #Total de reglas encontradas 


# In[10]:


ResultadosC1


# In[11]:


df = pd.DataFrame(ResultadosC1)
df


# In[12]:


print(ResultadosC1[8])


# La octava regla contiene los elementos **Atypical**, **Mr. Robot** y **Sex Education** que se ven juntos.
# 
# * La serie Atypical es una comedia de drama igual que Sex Education, sin embargo Mr. Robot es de tipo thriller.
# 
# * Podemos ver que tenemos 3 casos, cuando se ve Atypical y Mr. Robot para después ver Sex Education, cuando se ve Atypical y Sex Education para después ver Mr. Robot, y cuando se ve Mr. Robot y Sex Education para después ver Atypical. Analizando la confidencia de estas subreglas, la más alta es la primera subregla.
# 
# * El soporte es de 0.015 (1.5%), la confianza de 0.5618 (56.18%), la elevación de 2.197, lo que significa que se tiene 2.197 veces más posibilidad de que vean Atypical y Mr. Robot juntos para después de ver Ozark.

# In[13]:


print(ResultadosC1[9])


# La novena regla contiene los elementos **Atypical**, **Sex Education** y **Ozark** que se ven juntos.
# 
# * La serie Atypical es una comedia de drama igual que Sex Education, sin embargo Ozark es de drama.
# 
# * Podemos ver que tenemos 3 casos, cuando se ve Atypical y Ozark para después ver Sex Education, cuando se ve Atypical y Sex Education para después ver Ozark, y cuando se ve Ozark y Sex Education para después ver Atypical. Analizando la confidencia de estas subreglas, la más alta es la primera subregla.
# 
# * El soporte es de 0.022 (2.2%), la confianza de 0.5118 (51.18%), la elevación de 2, lo que significa que se tiene 2 veces más posibilidad de que vean Atypical y Ozark juntos para después de ver Sex Education.

# In[14]:


print(ResultadosC1[10])


# La novena regla contiene los elementos **Atypical**, **The Blacklist** y **Sex Education** que se ven juntos.
# 
# * La serie Atypical es una comedia de drama igual que Sex Education, sin embargo The Blacklist es de drama criminal.
# 
# * Podemos ver que tenemos 3 casos, cuando se ve Atypical y Sex Education para después ver The Blacklist, cuando se ve Atypical y The Blacklist para después ver Sex Education, y cuando se ve The Blacklist y Sex Education para después ver Atypical. Analizando la confidencia de estas subreglas, la más alta es la segunda subregla.
# 
# * El soporte es de 0.015 (1.5%), la confianza de 0.5191 (51.91%), la elevación de 2.03, lo que significa que se tiene 2.03 veces más posibilidad de que vean Atypical y The Blacklist juntos para después de ver Sex Education.

# In[15]:


for item in ResultadosC1:
  #El primer índice de la lista
  Emparejar = item[0]
  items = [x for x in Emparejar]
  print("Regla: " + str(item[0]))

  #El segundo índice de la lista
  print("Soporte: " + str(item[1]))

  #El tercer índice de la lista
  print("Confianza: " + str(item[2][0][2]))
  print("Lift: " + str(item[2][0][3])) 
  print("=====================================") 


# **Configuración 2**
# 
# Obtener reglas para aquellos shows que se hayan visto 30 veces al día (210 a la semana), entonces:
# 
# i) El soporte mínimo se calcula de 210/9690 = 0.02167 (2.167%).
# 
# ii) La confianza mínima para las reglas de 30%.
# 
# iii) La elevación igual a 2.

# In[16]:


sup = 210/9690
confi = 0.30
lifto = 2
ReglasC2 = apriori(TransaccionesLista, 
                   min_support=sup, 
                   min_confidence=confi, 
                   min_lift=lifto)


# In[17]:


ResultadosC2 = list(ReglasC2)
print(len(ResultadosC2))


# In[18]:


ResultadosC2 


# In[19]:


print(ResultadosC2[0])


# La primera regla contiene dos elementos: **Family Guy** y **Ozark** que se ven juntos.
# 
# * Family Guy es una serie animada que hace parodia de una familia estadounidense, mientras que Ozark es una serie de drama. Pudiera ser el caso de personas viviendo juntas que usan la misma cuenta durante el día. Que una persona elija qué se ve primero y después otra persona elija qué se ve después.
# 
# * El soporte es de 0.029 (2.9%), la confianza de 0.4069 (40.69%), la elevación de 2.1, esto es, hay casi 2.1 veces más probabilidades de que después de ver Family Guy, se pongan a ver Ozark.

# In[20]:


print(ResultadosC2[1])


# La segunda regla contiene dos elementos: **Mr. Robot** y **Ozark** que se ven juntos.
# 
# * Mr. Robot es una serie de drama igual que Ozark. Tiene sentido que se vean juntas al ser del mismo tipo.
# 
# * El soporte es de 0.0476 (4.76%), la confianza de 0.4349 (43.49%), la elevación de 2.25, esto es, hay casi 2.25 veces más probabilidades de ver Mr. Robot y luego ver Ozark. 

# In[21]:


print(ResultadosC2[3])


# La tercera regla contiene los elementos: **Mr. Robot**, **Ozark** y **Sex Education** que se ven juntos.
# 
# * Mr. Robot es una serie de drama igual que Ozark y Sex Education es una comedia drama. Se tienen dos casos, cuando se ven Mr. Robot y Sex Education para luego ver Ozark, y cuando se ve Ozark y Sex Education para luego ver Mr. Robot. La primer subregla es la que tiene una confianza mayor.
# 
# * El soporte es de 0.0231 (2.31%), la confianza de 0.4726 (47.26%), la elevación de 2.44, esto es, hay casi 2.44 veces más probabilidades de ver Mr. Robot y Sex Education para luego ver Ozark. 

# In[22]:


for item in ResultadosC2:
  #El primer índice de la lista
  Emparejar = item[0]
  items = [x for x in Emparejar]
  print("Regla: " + str(item[0]))

  #El segundo índice de la lista
  print("Soporte: " + str(item[1]))

  #El tercer índice de la lista
  print("Confianza: " + str(item[2][0][2]))
  print("Lift: " + str(item[2][0][3])) 
  print("=====================================") 


# **Configuración 3**
# 
# Obtener reglas para aquellos shows que se hayan visto 40 veces al día (280 a la semana), entonces:
# 
# i) El soporte mínimo se calcula de 280/9680 = 0.0289 (2.89%).
# 
# ii) La confianza mínima para las reglas de 20%.
# 
# iii) La elevación igual a 1.8.

# In[37]:


# sup = 200/9690
sup = .0206
confi = 0.20
lifto = 1.8
ReglasC3 = apriori(TransaccionesLista, 
                   min_support=sup, 
                   min_confidence=confi, 
                   min_lift=lifto)


# In[38]:


ResultadosC3 = list(ReglasC3)
print(len(ResultadosC3))


# In[39]:


ResultadosC3


# In[40]:


print(ResultadosC3[0])


# La primera regla contiene dos elementos: **Atypical** y **The Blacklist** que se ven juntos.
# 
# * Atypical es una comedia de drama, mientras que The Blacklist es una serie de drama criminal. Se tienen dos escenarios, primero ver Atypical y luego The Blacklist, o ver primero The Blacklist y luego Atypical. El segundo caso tiene una confianza mayor.
# 
# * El soporte es de 0.0296 (2.96%), la confianza de 0.2825 (28.25%), la elevación de 2.02, esto es, hay 2.02 veces más probabilidades de que después de ver The Blacklist, se vea Atypical.

# In[41]:


print(ResultadosC3[1])


# La segunda regla contiene dos elementos: **Dr. House** y **Sex Education** que se ven juntos.
# 
# * Dr. House es una serie de drama médico y Sex Education es una comedia de drama, ambas series son medianamente educativas, lo que me hace sentido que las vean juntas.
# 
# * El soporte es de 0.03 (3%), la confianza de 0.474 (47.4%), la elevación de 1.85, esto es, hay 1.85 veces más probabilidades de que después de ver Dr. House, se vea Sex Education.

# In[42]:


print(ResultadosC3[3])


# La primera regla contiene dos elementos: **Mr. Robot** y **Ozark** que se ven juntos.
# 
# * Mr. Robot es una serie de drama igual que Ozark. Se tienen dos escenarios, primero ver Mr. Robot y luego Ozark, o ver primero Ozark y luego Mr. Robot. El primer caso tiene una confianza mayor.
# 
# * El soporte es de 0.048 (4.8%), la confianza de 0.4349 (43.49%), la elevación de 2.25, esto es, hay 2.25 veces más probabilidades de que después de ver Mr. Robot, se vea Ozark.

# In[43]:


for item in ResultadosC3:
  #El primer índice de la lista
  Emparejar = item[0]
  items = [x for x in Emparejar]
  print("Regla: " + str(item[0]))

  #El segundo índice de la lista
  print("Soporte: " + str(item[1]))

  #El tercer índice de la lista
  print("Confianza: " + str(item[2][0][2]))
  print("Lift: " + str(item[2][0][3])) 
  print("=====================================") 

