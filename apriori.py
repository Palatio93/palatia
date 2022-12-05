import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori

def datos_transacciones():
  DatosTransacciones = pd.read_csv('tv_shows_dataset.csv', header=None)
  return DatosTransacciones

def transacciones_a_listas():
  DatosTransacciones = datos_transacciones()
  Transacciones = DatosTransacciones.values.reshape(-1).tolist()
  return Transacciones

def transacciones_a_matriz():
  Transacciones = transacciones_a_listas()
  Lista = pd.DataFrame(Transacciones)
  Lista['Frecuencia'] = 1
  Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
  Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
  Lista = Lista.rename(columns={0 : 'Item'})
  return Lista

def crea_grafico():
  Lista = transacciones_a_matriz()
  plt.figure(figsize=(16,20), dpi=300)
  plt.ylabel('Item')
  plt.xlabel('Frecuencia')
  plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
  plt.savefig('fig1.png')
  plt.savefig('fig1.pdf')

def sanitization():
  DatosTransacciones = datos_transacciones()
  TransaccionesLista = DatosTransacciones.stack().groupby(level=0).apply(list).tolist()
  return TransaccionesLista 

def aplica_algo(sup, confi, lifto):
  TransaccionesLista = sanitization()
  ReglasC1 = apriori(TransaccionesLista, 
                    min_support=sup, 
                    min_confidence=confi, 
                    min_lift=lifto)
  ResultadosC1 = list(ReglasC1)
  #print(len(ResultadosC1)) #Total de reglas encontradas
  #df = pd.DataFrame(ResultadosC1)
  return ResultadosC1

def muestra_resultados(soporte,confianza,lifto):
  Resultados = aplica_algo(soporte,confianza,lifto)
  reglas = [{
    "Regla": ", ".join([x for x in res[0]]),
    "Soporte":res[1]*100,
    "Confianza":res[2][0][2]*100,
    "Lift":res[2][0][3]
  } for res in Resultados]
  # for item in Resultados:
  #   #El primer índice de la lista
  #   Emparejar = item[0]
  #   items = [x for x in Emparejar]
  #   print("Regla: " + str(item[0]))

  #   #El segundo índice de la lista
  #   print("Soporte: " + str(item[1]))

  #   #El tercer índice de la lista
  #   print("Confianza: " + str(item[2][0][2]))
  #   print("Lift: " + str(item[2][0][3])) 
  #   print("=====================================")
  return reglas

def main(soporte,confianza,lifto):
  print("Leyendo dataset")
  # print(datos_transacciones())
  print("Pasando a listas")
  # print(transacciones_a_listas())
  print("Pasando a matriz")
  # print(transacciones_a_matriz())
  print("Creando grafico")
  crea_grafico()
  print("Sanitizando")
  # print(sanitization())
  print("Aplicando algoritmo apriori")
  #print(aplica_algo())
  print("Muestra de resultados")
  Resultados = muestra_resultados(soporte,confianza,lifto)

  return Resultados