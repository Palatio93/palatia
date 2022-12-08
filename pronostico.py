import pandas as pd                         
import numpy as np                          
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
import yfinance as yf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

def get_dataset():
  DataElektra = yf.Ticker('ELEKTRA.MX')
  return DataElektra

def get_hist(DataElektra):
  ElektraHist = DataElektra.history(start = '2019-1-1', end = '2022-11-27', interval='1d')
  return ElektraHist

def model1(ElektraHist):
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
  plt.savefig('./static/images/pronostico/pronostico1.png')
  plt.savefig('./static/images/pronostico/pronostico1.pdf')

  MDatos1 = ElektraHist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
  MDatos1 = MDatos1.dropna()

  X = np.array(MDatos1[['Open',
                      'High',
                      'Low']])

  Y = np.array(MDatos1[['Close']])

  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                      test_size = 0.2, 
                                                                      random_state = 0, 
                                                                      shuffle = True)


  RLMultiple = linear_model.LinearRegression()
  RLMultiple.fit(X_train, Y_train)

  Y_Pronostico = RLMultiple.predict(X_test)

  r2_score(Y_test, Y_Pronostico)

  print('Coeficientes: \n', RLMultiple.coef_)
  print('Intercepto: \n', RLMultiple.intercept_)
  print("Residuo: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
  print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
  print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
  print('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))


  plt.figure(figsize=(20, 5))
  plt.plot(Y_test, color='red', marker='+', label='Real')
  plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
  plt.xlabel('Fecha')
  plt.ylabel('Precio de las acciones')
  plt.title('Pronóstico de las acciones de Elektra')
  plt.grid(True)
  plt.legend()
  plt.savefig('./static/images/pronostico/pronostico2.png')
  plt.savefig('./static/images/pronostico/pronostico2.pdf')

  PrecioAccion = pd.DataFrame({'Open': [995.010010],
                              'High': [1024.890015], 
                              'Low': [995.010010]})
  Precio1 = RLMultiple.predict(PrecioAccion)
  return Precio1

def model2(ElektraHist):
  MDatos2 = ElektraHist.drop(columns = ['Dividends', 'Stock Splits'])
  MDatos2 = MDatos2.dropna()
  X = np.array(MDatos2[['Open',
                      'High',
                      'Low',
                      'Volume']])

  Y = np.array(MDatos2[['Close']])

  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                      test_size = 0.2, 
                                                                      random_state = 0, 
                                                                      shuffle = True)

  RLMultiple = linear_model.LinearRegression()
  RLMultiple.fit(X_train, Y_train)

  Y_Pronostico = RLMultiple.predict(X_test)

  r2_score(Y_test, Y_Pronostico)

  print('Coeficientes: \n', RLMultiple.coef_)
  print('Intercepto: \n', RLMultiple.intercept_)
  print("Residuo: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
  print("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
  print("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
  print('Score (Bondad de ajuste): %.4f' % r2_score(Y_test, Y_Pronostico))

  plt.figure(figsize=(20, 5))
  plt.plot(Y_test, color='red', marker='+', label='Real')
  plt.plot(Y_Pronostico, color='green', marker='+', label='Estimado')
  plt.xlabel('Fecha')
  plt.ylabel('Precio de las acciones')
  plt.title('Pronóstico de las acciones de Elektra')
  plt.grid(True)
  plt.legend()
  plt.savefig('./static/images/pronostico/pronostico3.png')
  plt.savefig('./static/images/pronostico/pronostico3.pdf')

  PrecioAccion2 = pd.DataFrame({'Open': [995.010010],
                              'High': [1024.890015], 
                              'Low': [995.010010],
                              'Volume':[110698]})
  Precio2 = RLMultiple.predict(PrecioAccion2)
  return Precio2


def main():
  DataElektra = get_dataset()
  ElektraHist = get_hist(DataElektra)
  model1(ElektraHist)
  model2(ElektraHist)
  return