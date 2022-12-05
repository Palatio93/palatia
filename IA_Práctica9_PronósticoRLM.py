#!/usr/bin/env python
# coding: utf-8

# # **Práctica 9: Pronóstico (Precio de las acciones)**
# 
# Nombre: Humberto Ignacio Hernández Olvera
# 
# No. Cuenta: 309068165
# 
# Email: humberto1nacho@gmail.com

# ### **Contexto**
# 
# Yahoo Finance ofrece una amplia variedad de datos de mercado sobre acciones, bonos, divisas y criptomonedas. También proporciona informes de noticias con varios puntos de vista sobre diferentes mercados de todo el mundo, todos accesibles a través de la biblioteca yfinance.
# 
# **Objetivo:** Generar un modelo de pronóstico de las acciones de una determinada empresa a través de un algoritmo de aprendizaje automático.
# 
# **Fuente de datos**
# 
# De Yahoo Finanzas se utiliza el Ticker -Etiqueta de cotización- de la acción bursatil.

# ### **Importar las bibliotecas y los datos**

# In[1]:


get_ipython().system('pip install yfinance')
#!pip install googlefinance


# In[2]:


import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf


# In[3]:


# Para Elektra
DataElektra = yf.Ticker('ELEKTRA.MX')
# Ticker - abreviacion, etiqueta de la data


# In[4]:


ElektraHist = DataElektra.history(start = '2019-1-1', end = '2022-11-27', interval='1d')
ElektraHist


# Descripción:
# 
# * En el comercio de acciones, 'alto' y 'bajo' se refieren a los precios máximos y mínimos en un período determinado.
# * 'Apertura' y 'cierre' son los precios en los que una acción comenzó y terminó cotizando en el mismo período. 
# * El 'volumen' es la cantidad total de la actividad comercial. 
# * Los valores ajustados tienen en cuenta las acciones corporativas, como los 'dividendos', la 'división de acciones' y la emisión de nuevas acciones.

# ### **Descripción de la estructura de los datos**

# Se puede usar **info()** para obtener el tipo de datos y la suma de valores nulos. Se observa que los datos son numéricos (flotante y entero).

# In[5]:


ElektraHist.info()


# In[6]:


ElektraHist.describe()


# * Se incluye un recuento, media, desviación, valor mínimo, valor máximo, percentil inferior (25%), 50% y percentil superior (75%).
# * Por defecto, el percentil 50 es lo mismo que la mediana.
# * Se observa que para cada variable, el recuento también ayuda a identificar variables con valores nulos o vacios. Estos son: **Dividends** y **Stock Splits**.

# ### **Gráfica de los precios de las acciones**

# In[7]:


plt.figure(figsize=(20, 5))
plt.plot(ElektraHist['Open'], color='purple', marker='+', label='Open')
plt.plot(ElektraHist['High'], color='blue', marker='+', label='High')
plt.plot(ElektraHist['Low'], color='orange', marker='+', label='Low')
plt.plot(ElektraHist['Close'], color='green', marker='+', label='Close')
plt.xlabel('Fecha')
plt.ylabel('Precio de las acciones')
plt.title('Elektra')
plt.grid(True)
plt.legend()
plt.show()


# In[8]:


MDatos1 = ElektraHist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
MDatos1


# In[9]:


# En caso de tener valores nulos
MDatos1 = MDatos1.dropna()
MDatos1


# ### Aplicación del algoritmo

# In[10]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection


# Se seleccionan las variables predictoras (X) y la variable a pronosticar (Y)

# In[11]:


X = np.array(MDatos1[['Open',
                     'High',
                     'Low']])
pd.DataFrame(X)


# In[12]:


Y = np.array(MDatos1[['Close']])
pd.DataFrame(Y)


# Se hace la división de los datos

# In[13]:


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)


# In[14]:


pd.DataFrame(X_test)


# Se entrena el modelo a través de Regresión Lineal Múltiple

# In[15]:


RLMultiple = linear_model.LinearRegression()
RLMultiple.fit(X_train, Y_train)


# In[16]:


#Se genera el pronóstico
Y_Pronostico = RLMultiple.predict(X_test)
pd.DataFrame(Y_Pronostico)


# In[17]:


r2_score(Y_test, Y_Pronostico)


# In[18]:


print('Coeficientes: \n', RLMultiple.coef_)
print('Intercepto: \n', RLMultiple.intercept_)
print("Residuo: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
print('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))


# #### **Conformación del modelo de pronóstico**
# 
# Y = 5.3312 - 0.022(Open) + 0.507(High) + 0.510(Low) + 6.3179
# 
# * Se tiene un Score de 0.9974, que indica que el pronóstico del precio de cierre de la acción se logrará con un 99.74% de efectividad.
# * Además, los pronósticos del modelo final se alejan en promedio 85.973 y 9.2722 unidades del valor real, esto es, MSE y RMSE, respectivamente.

# In[19]:


plt.figure(figsize=(20, 5))
plt.plot(Y_test, color='red', marker='+', label='Real')
plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
plt.xlabel('Fecha')
plt.ylabel('Precio de las acciones')
plt.title('Pronóstico de las acciones de Elektra')
plt.grid(True)
plt.legend()
plt.show()


# #### **Nuevos pronósticos**

# In[20]:


PrecioAccion = pd.DataFrame({'Open': [995.010010],
                             'High': [1024.890015], 
                             'Low': [995.010010]})
Precio1 = RLMultiple.predict(PrecioAccion)
Precio1


# 
# ---
# 
# ## Inicia la nueva matriz 2

# In[21]:


MDatos2 = ElektraHist.drop(columns = ['Dividends', 'Stock Splits'])
MDatos2


# In[22]:


# En caso de tener valores nulos
MDatos2 = MDatos2.dropna()
MDatos2


# In[23]:


X = np.array(MDatos2[['Open',
                     'High',
                     'Low',
                    'Volume']])
pd.DataFrame(X)


# In[24]:


Y = np.array(MDatos2[['Close']])
pd.DataFrame(Y)


# In[25]:


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)


# In[26]:


pd.DataFrame(X_test)


# In[27]:


RLMultiple = linear_model.LinearRegression()
RLMultiple.fit(X_train, Y_train)


# In[28]:


#Se genera el pronóstico
Y_Pronostico = RLMultiple.predict(X_test)
pd.DataFrame(Y_Pronostico)


# In[29]:


r2_score(Y_test, Y_Pronostico)


# In[30]:


print('Coeficientes: \n', RLMultiple.coef_)
print('Intercepto: \n', RLMultiple.intercept_)
print("Residuo: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
print('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))


# #### **Conformación del modelo de pronóstico**
# 
# Y = 4.6891 - 3.2084e-02(Open) + 5.0233e-01(High) + 5.2613e-01(Low) + 3.8427e-06(Volume) + 5.4436
# 
# * Se tiene un Score de 0.9979, que indica que el pronóstico del precio de cierre de la acción se logrará con un 99.79% de efectividad.
# * Además, los pronósticos del modelo final se alejan en promedio 68.84 y 8.297 unidades del valor real, esto es, MSE y RMSE, respectivamente.

# In[31]:


plt.figure(figsize=(20, 5))
plt.plot(Y_test, color='red', marker='+', label='Real')
plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
plt.xlabel('Fecha')
plt.ylabel('Precio de las acciones')
plt.title('Pronóstico de las acciones de Elektra')
plt.grid(True)
plt.legend()
plt.show()


# **Nuevos pronósticos**

# In[32]:


PrecioAccion2 = pd.DataFrame({'Open': [995.010010],
                             'High': [1024.890015], 
                             'Low': [995.010010],
                             'Volume':[110698]})
Precio2 = RLMultiple.predict(PrecioAccion2)
Precio2


# In[33]:


Precio1

