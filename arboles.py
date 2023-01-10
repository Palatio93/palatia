import pandas as pd            
import numpy as np              
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_text

def get_data(path=None):
  Covid = pd.read_csv('CovidAdultosMayores.csv')
  return Covid

def get_heatmap(Covid):
  plt.figure(figsize=(14,7))
  MatrizInf = np.triu(Covid.corr())
  sns.heatmap(Covid.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
  plt.savefig('./static/images/arboles/arboles1.png')
  plt.savefig('./static/images/arboles/arboles1.pdf')

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

def print_tree(ClasificacionAD):
  plt.figure(figsize=(16,16))  
  plot_tree(ClasificacionAD, 
            feature_names = ['SEXO',
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
                            'UCI'],
          class_names = ['1', '2', '3'])
  plt.savefig('./static/images/arboles/arboles2.png')
  plt.savefig('./static/images/arboles/arboles2.pdf')

def arbol(X_train, X_validation, Y_train, Y_validation, Covid, print_tree1=False):
  ClasificacionAD = DecisionTreeClassifier(random_state=0)
  ClasificacionAD.fit(X_train, Y_train)
  Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
  ValoresMod1 = pd.DataFrame(Y_validation, Y_ClasificacionAD)
  acc_score = accuracy_score(Y_validation, Y_ClasificacionAD)
  ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
  Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                    ModeloClasificacion1, 
                                    rownames=['Actual'], 
                                    colnames=['Clasificación']) 
  # print('Criterio: \n', ClasificacionAD.criterion)
  # print('Importancia variables: \n', ClasificacionAD.feature_importances_)
  # print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionAD))
  # print(classification_report(Y_validation, Y_ClasificacionAD))
  criterio = ClasificacionAD.criterion
  reporte_clasif = classification_report(Y_validation, Y_ClasificacionAD)
  ImportanciaMod1 = pd.DataFrame({'Variable': list(Covid[['SEXO',
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
                                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
  # print(ImportanciaMod1)
  # Reporte = export_text(ClasificacionAD, feature_names = ['SEXO',
  #                         'TIPO_PACIENTE',
  #                         'INTUBADO',
  #                         'NEUMONIA',
  #                         'EDAD',
  #                         'DIABETES',
  #                         'EPOC',
  #                         'ASMA',
  #                         'INMUSUPR',
  #                         'HIPERTENSION',
  #                         'OTRA_COM',
  #                         'CARDIOVASCULAR',
  #                         'OBESIDAD',
  #                         'RENAL_CRONICA',
  #                         'TABAQUISMO', 
  #                         'OTRO_CASO',
  #                         'RESULTADO_ANTIGENO',
  #                         'CLASIFICACION_FINAL',
  #                         'UCI'])
  if print_tree1:
    print_tree(ClasificacionAD)
  return acc_score, Matriz_Clasificacion1, criterio, reporte_clasif, ImportanciaMod1



def main(print_tree1=False):
  Covid = get_data()
  get_heatmap(Covid)
  X = get_variables_predict(Covid)
  Y = get_variable_clase(Covid)
  X_train, X_validation, Y_train, Y_validation = making_models(X,Y)
  acc_score, Matriz_Clasificacion1, criterio, reporte_clasif, ImportanciaMod1 = arbol(X_train, X_validation, Y_train, Y_validation, Covid, print_tree1)
  acc_score = round(acc_score*100,3)
  Matriz_Clasificacion = [{
    "Actual": index,
    "Clasificacion": {
      "Finado":Matriz_Clasificacion1["Finado"][index],
      "Vivo":Matriz_Clasificacion1["Vivo"][index]
    }
  } for index in Matriz_Clasificacion1]
  # print(Matriz_Clasificacion)
  # print(Matriz_Clasificacion1)
  # print(Matriz_Clasificacion1)
  # print("###########")
  # print(Matriz_Clasificacion)
  # print(ImportanciaMod1)
  # print(type(ImportanciaMod1))
  ImportanciaMod = [{
    "Variable": ImportanciaMod1["Variable"][index],
    "Valor": round(ImportanciaMod1["Importancia"][index],3)
  } for index in ImportanciaMod1.index]
  reporte_clasif = reporte_clasif.split(None)
  reporte_clasif2 = "Se tiene una precisión para Finado de {pre1} y recall de {rec1}, mientras que para Vivo se tiene una precisión de {pre2} y recall de {rec2}".format(pre1=reporte_clasif[5], rec1=reporte_clasif[6], pre2=reporte_clasif[10], rec2=reporte_clasif[11])
  return acc_score, Matriz_Clasificacion, criterio, reporte_clasif2, ImportanciaMod

if __name__ == "__main__":
  main()