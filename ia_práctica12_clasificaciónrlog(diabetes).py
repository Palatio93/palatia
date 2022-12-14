# -*- coding: utf-8 -*-
"""IA_Práctica12-ClasificaciónRLog(Diabetes).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lBmDK27s-N6EdOhghhy556FF91-f1CI7

## **Práctica 12: Clasificación (Diabétes)**

Nombre:

No. Cuenta:

Email:

### **Caso de estudio**

Estudios clínicos de diabetes en una población femenina.

**Objetivo.** Clasificar si una persona tiene diabetes o no, en función de otros parámetros disponibles, como número de embarazos, glucosa, presión arterial, índice de masa corporal, los niveles de insulina, entre otros. Este es un problema de clasificación y se requiere obtener el mejor modelo de aprendizaje automático para predecir la diabetes.

**Emplear los algoritmos:** 

* Regresión logística.
* Árbol de decisión.
* Bosque aleatorio.
* Máquinas de soporte vectorial.

**Fuente de datos:**

https://www.kaggle.com/saurabh00007/diabetescsv

**Variables:**

* Número de embarazos (Pregnancies): número de veces que ha estado embarazada la persona.

* Concentración de glucosa en plasma (Glucose): cantidad de glucosa en la sangre. Cuando una persona ha ingerido alimento los valores normales son menores a 140 mg/DL y cuando los resultados se dan entre 140 a 190 son indicativos de diabetes.

* Presión arterial diastólica (BloodPressure): es la cantidad de presión que hay en las arterias ente un latido y otro.

* Espesor del pliegue cutáneo (SkinThickness): es un procedimiento frecuentemente utilizado, en combinación con el índice de masa corporal (IMC), para estimar la grasa corporal. Medir los pliegues cutáneos permite valorar los depósitos de grasa del cuerpo humano. A modo de referencia, según la medicina el espesor normal: ♂ 12 mm; ♀ 23 mm.

* Insulina (Insulin): es una prueba de insulina que consiste analizar antes de administrar la glucosa y 2 horas después. La razón por la que se realizan estas pruebas es para ver la curva de respuesta a la glucosa.

* Índice de masa corporal (BMI): es utilizado para estimar la cantidad de grasa corporal, y determinar si el peso está dentro del rango normal, o por el contrario, se tiene sobrepeso o delgadez.

* Función pedigrí de la diabetes (DiabetesPedigreeFunction): es una función que califica la probabilidad de diabetes según los antecedentes familiares.

* Edad en años (Age).

* Resultado (Outcome): si es positivo o negativo al diagnóstico de diabetes.

### **I. Acceso a datos y selección de características**

#### **1) Acceso a los datos**
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
# %matplotlib inline

Diabetes = pd.read_csv('Datos/Diabetes.csv')
Diabetes

Diabetes.info()

print(Diabetes.groupby('Outcome').size())

plt.figure(figsize=(10, 7))
plt.scatter(Diabetes['BloodPressure'], Diabetes['Glucose'], c = Diabetes.Outcome)
plt.grid()
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.show()

Diabetes.describe()

"""#### **2) Selección de características**

A través de un mapa de calor de identifican posibles variables correlacionadas.
"""

plt.figure(figsize=(14,7))
MatrizInf = np.triu(Diabetes.corr())
sns.heatmap(Diabetes.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
plt.show()

"""**Varibles seleccionadas:**

Ante la no presencia de correlaciones altas (fuertes), se consideran a todas las variables para la construcción de los modelos.

#### **3) Definición de las variables predictoras y variable clase**
"""

#Variables predictoras
X = np.array(Diabetes[['Pregnancies', 
                       'Glucose', 
                       'BloodPressure', 
                       'SkinThickness', 
                       'Insulin', 
                       'BMI',
                       'DiabetesPedigreeFunction',
                       'Age']])
pd.DataFrame(X)

#Variable clase
Y = np.array(Diabetes[['Outcome']])
pd.DataFrame(Y)

"""### **II. Creación de los modelos**


#### **Modelo 1: Regresión Logística**
"""

from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

print(len(X_train))
print(len(X_validation))

#Se entrena el modelo a partir de los datos de entrada
ClasificacionRL = linear_model.LogisticRegression()
ClasificacionRL.fit(X_train, Y_train)

#Predicciones probabilísticas
Probabilidad = ClasificacionRL.predict_proba(X_validation)
pd.DataFrame(Probabilidad)

#Clasificación final 
Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
print(Y_ClasificacionRL)

accuracy_score(Y_validation, Y_ClasificacionRL)

"""### **III. Validación**

#### **Validación: Regresión Logística**
"""

#Matriz de clasificación
ModeloClasificacion = ClasificacionRL.predict(X_validation)
Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                   ModeloClasificacion, 
                                   rownames=['Reales'], 
                                   colnames=['Clasificación']) 
Matriz_Clasificacion

#Reporte de la clasificación
print("Exactitud:", accuracy_score(Y_validation, Y_ClasificacionRL))
print(classification_report(Y_validation, Y_ClasificacionRL))

from sklearn.metrics import RocCurveDisplay

CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation, name="Diabetes")
plt.show()

"""### **IV. Nuevas clasificaciones**

"""

#Paciente
PacienteID = pd.DataFrame({'Pregnancies': [6],
                           'Glucose': [148],
                           'BloodPressure': [72],
                           'SkinThickness': [35],
                           'Insulin': [0],
                           'BMI': [33.6],
                           'DiabetesPedigreeFunction': [0.627],
                           'Age': [50]})
ClasificacionRL.predict(PacienteID)

