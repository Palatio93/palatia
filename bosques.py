import pandas as pd            
import numpy as np              
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def get_data(path=None):
  Covid = pd.read_csv('CovidAdultosMayores.csv')
  return Covid

def get_heatmap(Covid):
  plt.figure(figsize=(14,7))
  MatrizInf = np.triu(Covid.corr())
  sns.heatmap(Covid.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
  plt.savefig('./static/images/bosques/bosques1.png')
  plt.savefig('./static/images/bosques/bosques1.pdf')

def get_variables_predict(Covid):
  X = np.array(Covid[['SEXO',
                          'TIPO_PACIENTE',
                          'INTUBADO',
                          'NEUMONIA',
                          'EDAD',
                          'DIABETES',
                          'EPOC',
                          'ASMA',
                          'INMUSUPR',
                          'HIPERTENSION',
                          'OTRA_COM',
                          'CARDIOVASCULAR',
                          'OBESIDAD',
                          'RENAL_CRONICA',
                          'TABAQUISMO', 
                          'OTRO_CASO',
                          'RESULTADO_ANTIGENO',
                          'CLASIFICACION_FINAL',
                          'UCI']])
  return X
  
def get_variable_clase(Covid):
  Y = np.array(Covid[['SITUACION']])
  return Y

def making_models(X,Y):
  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
  return X_train, X_validation, Y_train, Y_validation

def bosques(X_train, X_validation, Y_train, Y_validation, Covid):
  ClasificacionBA = RandomForestClassifier(random_state=0)
  ClasificacionBA.fit(X_train, Y_train)
  Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
  ValoresMod2 = pd.DataFrame(Y_validation, Y_ClasificacionBA)
  acc_score = accuracy_score(Y_validation, Y_ClasificacionBA)
  ModeloClasificacion2 = ClasificacionBA.predict(X_validation)
  Matriz_Clasificacion2 = pd.crosstab(Y_validation.ravel(),
                                      ModeloClasificacion2,
                                      rownames=['Reales'],
                                      colnames=['Clasificación']) 
  # print('Criterio: \n', ClasificacionBA.criterion)
  # print('Importancia variables: \n', ClasificacionBA.feature_importances_)
  # print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionBA))
  # print(classification_report(Y_validation, Y_ClasificacionBA))
  criterio = ClasificacionBA.criterion
  reporte_clasif = classification_report(Y_validation, Y_ClasificacionBA)
  Importancia2 = pd.DataFrame({'Variable': list(Covid[['SEXO',
                          'TIPO_PACIENTE',
                          'INTUBADO',
                          'NEUMONIA',
                          'EDAD',
                          'DIABETES',
                          'EPOC',
                          'ASMA',
                          'INMUSUPR',
                          'HIPERTENSION',
                          'OTRA_COM',
                          'CARDIOVASCULAR',
                          'OBESIDAD',
                          'RENAL_CRONICA',
                          'TABAQUISMO', 
                          'OTRO_CASO',
                          'RESULTADO_ANTIGENO',
                          'CLASIFICACION_FINAL',
                          'UCI']]), 
                             'Importancia': ClasificacionBA.feature_importances_}).sort_values('Importancia', ascending=False)
  # print(Importancia2)
  return acc_score, Matriz_Clasificacion2, criterio, reporte_clasif, Importancia2

def main():
  Covid = get_data()
  get_heatmap(Covid)
  X = get_variables_predict(Covid)
  Y = get_variable_clase(Covid)
  X_train, X_validation, Y_train, Y_validation = making_models(X,Y)
  acc_score, Matriz_Clasificacion2, criterio, reporte_clasif, ImportanciaMod2 = bosques(X_train, X_validation, Y_train, Y_validation, Covid)
  acc_score = round(acc_score*100,3)
  Matriz_Clasificacion = [{
    "Actual": index,
    "Clasificacion": {
      "Finado":Matriz_Clasificacion2["Finado"][index],
      "Vivo":Matriz_Clasificacion2["Vivo"][index]
    }
  } for index in Matriz_Clasificacion2]
  ImportanciaMod = [{
    "Variable": ImportanciaMod2["Variable"][index],
    "Valor": round(ImportanciaMod2["Importancia"][index],3)
  } for index in ImportanciaMod2.index]
  reporte_clasif = reporte_clasif.split(None)
  reporte_clasif2 = "Se tiene una precisión para Finado de {pre1} y recall de {rec1}, mientras que para Vivo se tiene una precisión de {pre2} y recall de {rec2}".format(pre1=reporte_clasif[5], rec1=reporte_clasif[6], pre2=reporte_clasif[10], rec2=reporte_clasif[11])
  # print(acc_score)
  # print(Matriz_Clasificacion2)
  return acc_score, Matriz_Clasificacion, criterio, reporte_clasif2, ImportanciaMod

