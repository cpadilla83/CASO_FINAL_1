# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:59:45 2024


@author: CarlosPadilla & XavierAsmal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos del CancerDatabase
data = pd.read_csv('titanic.csv')


########## Entendimiento de la data ##########

#Verifica la cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(data.shape)

#Verifica el tipo de datos contenida en ambos dataset
print('Tipos de datos:')
print(data.info())

#Verifica los datos faltantes de los dataset
print('Datos faltantes:')
print(pd.isnull(data).sum())

#Verifica las estadísticas básicas del dataset
print('Estadísticas del dataset:')
print(data.describe())

########## Preprocesamiento de la data ##########

# Transforma los datos de la variable sexo (categórico) en números
data['Sex'].replace(['female','male'],[0,1],inplace=True)

#Transforma los datos de embarque (categórico) en números
data['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

#Reemplazo los datos faltantes en la edad por la media de esta variable
print(data["Age"].mean())
promedio = 30
data['Age'] = data['Age'].replace(np.nan, promedio)

#Crea varios grupos/rangos de edades
#Rangos: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
data['Age'] = pd.cut(data['Age'], bins, labels = names)


#Se elimina la columna de "Cabin" ya que tiene muchos datos perdidos
# El parámetro axis=1 indica que se deben eliminar columnas en lugar de filas (axis=0).
# El parámetro inplace indica si la operación se realiza directamente en el 
# DataFrame original o devolvuelve una nueva copia con las filas o columnas eliminadas.
data.drop(['Cabin'], axis = 1, inplace=True)

#Elimina las columnas que se considera que no son necesarias para el analisis
data = data.drop(['PassengerId','Name','Ticket'], axis=1)

#Se elimina las filas con datos perdidos
data.dropna(axis=0, how='any', inplace=True)

#Verifica los datos
print(pd.isnull(data).sum())

print(data.shape)

print(data.head())

# Guardar el DataFrame en un archivo CSV
# El parámetro index=False evita que los índices del DataFrame
# se guarden como una columna en el archivo CSV
data.to_csv('train_procesado.csv', index=False, sep=',', encoding='utf-8')


# 1. Histograma de la distribución de edades
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=7, kde=False)
plt.title('Distribución de Edades')
plt.xlabel('Rango de Edad')
plt.ylabel('Frecuencia')
plt.show()

# 2. Barplot del número de sobrevivientes por género
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Número de Sobrevivientes por Género')
plt.xlabel('Sobrevivió')
plt.ylabel('Frecuencia')
plt.legend(['Mujer', 'Hombre'])
plt.show()

# 3. Boxplot de tarifas por clase de pasajero
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=data)
plt.title('Distribución de Tarifas por Clase de Pasajero')
plt.xlabel('Clase de Pasajero')
plt.ylabel('Tarifa')
plt.show()

# 4. Countplot del puerto de embarque
plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', data=data)
plt.title('Frecuencia de Pasajeros por Puerto de Embarque')
plt.xlabel('Puerto de Embarque')
plt.ylabel('Frecuencia')
plt.xticks(ticks=[0, 1, 2], labels=['Queenstown', 'Southampton', 'Cherbourg'])
plt.show()

# 5. Matriz de correlación
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()


# Calcular la correlación de todas las variables con 'diagnosis'
correlation_with_target = data.corr()['Survived'].sort_values(ascending=False)

# Seleccionar las 10 variables más correlacionadas con 'diagnosis'
top_5_features = correlation_with_target.head(6).index.tolist()  # Incluye 'diagnosis'
top_5_features

# Generar la matriz de correlación solo con las variables más significativas
top_5_corr_matrix = data[top_5_features].corr()
# crea una máscara para ocultar la parte superior de la matriz de correlación
# con k=0 no incluye la diagonal principal y con k=1 si
mask = np.triu(np.ones_like(top_5_corr_matrix, dtype=bool), k=1)

# Crear un mapa de calor de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(top_5_corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación entre Variables Más significativas')
plt.show()

# Aplicar una máscara para mostrar solo correlaciones moderadas/altas mayores a 0.4
mask = np.abs(top_5_corr_matrix) < 0.4
top_5_corr_matrix[mask] = np.nan
# Crear un mapa de calor de correlación con valores significativos
plt.figure(figsize=(10, 8))
sns.heatmap(top_5_corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación (moderadas / altas)')
plt.show()




#MODELOS DE CLASIFICACIÓN
#LOGISTIC REGRESSION


# Logistic regression for breast cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Dividir en características (X) y objetivo (y)
X = data.drop('Survived', axis=1)
y = data['Survived']


# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para que todas las características tengan una escala similar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea y entrena el modelo de regresión logistica
model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=100)
model.fit(X_train, y_train)

# Imprime los coeficientes y el intercepto del modelo entrenado
print("\nCoeficientes del modelo:")
print(model.coef_)
print("\nIntercepto del modelo:")
print(model.intercept_)


# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)


# Convierte las probabilidades en etiquetas binarias (0 o 1)
y_pred = (y_pred > 0.5)

# Muestra el informe de evaluación del modelo entrenado
print(classification_report(y_test, y_pred))

# Matriz de confusión:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)
# gráfica cm
plt.figure(figsize = (8,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction', fontsize = 12)
plt.ylabel('Real', fontsize = 12)
plt.show()

# Exactitud:
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("accuracy: ", acc)

# Sensibilidad:
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("recall: ", recall)

# Precisión:
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("precision: ", precision)

# Especificidad
# 'specificity' is just a special case of 'recall'. 
# specificity is the recall of the negative class
specificity = recall_score(y_test, y_pred, pos_label=0)
print("specificity: ", specificity)

# Puntuación F1:
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("f1 score: ", f1)

# Área bajo la curva:
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print("auc: ", auc)

# Curva ROC
from sklearn.metrics import roc_curve
plt.figure()
lw = 2
plt.plot(roc_curve(y_test, y_pred)[0], roc_curve(y_test, y_pred)[1], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# R Score (R^2 coefficient of determination)
from sklearn.metrics import r2_score
R = r2_score(y_test, y_pred)
print("R2: ", R)

# Obtener los coeficientes del modelo
coefficients = model.coef_[0]
feature_names = X.columns

# Crear un DataFrame para visualizar los coeficientes
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

# Configurar el gráfico de barras
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='b')
ax.set_xlabel('Coefficient')
ax.set_title('Feature Importance')
plt.show()


# Guardar el modelo a un archivo
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')
# Cargar el modelo desde el archivo
loaded_model = joblib.load('logistic_regression_model.pkl')
# Hacer predicciones con el modelo cargado
y_pred = model.predict(X_test)


#KNN
#ELBOW


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset
# Dividir en características (X) y objetivo (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el rango de valores de k a evaluar
n = 21
k_range = range(1, n, 2) # en saltos de 2 (solo impares)
error_rates = []

# Evaluar el modelo para cada valor de k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error)


# Graficar la tasa de error para cada valor de k
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rates, marker='o', linestyle='--', color='b')
plt.title('Elbow method for selecting k in k-NN')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error Rate: (1- accuracy)')
plt.xticks(np.arange(1, n, 1))
plt.grid()
plt.show()



# K-NN for breast cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Carga el conjunto de datos Breast Cancer
# Dividir en características (X) y objetivo (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para que todas las características tengan una escala similar
scaler = MinMaxScaler(feature_range=(0,1)) # [0, 1]
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Crea y entrena el modelo K-NN
model = KNeighborsClassifier(n_neighbors=9, p=2,  # Función euclidean
                             weights='uniform')

model.fit(X_train, y_train)

# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)

# Convierte las probabilidades en etiquetas binarias (0 o 1)
# y_pred = (y_pred > 0.5)

# Muestra el informe de evaluación del modelo entrenado
print(classification_report(y_test, y_pred))

# Matriz de confusión:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)
# gráfica cm
plt.figure(figsize = (8,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction', fontsize = 12)
plt.ylabel('Real', fontsize = 12)
plt.show()

# Exactitud:
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("accuracy: ", acc)

# Sensibilidad:
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("recall: ", recall)

# Precisión:
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("precision: ", precision)

# Especificidad
# 'specificity' is just a special case of 'recall'. 
# specificity is the recall of the negative class
specificity = recall_score(y_test, y_pred, pos_label=0)
print("specificity: ", specificity)

# Puntuación F1:
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("f1 score: ", f1)

# Área bajo la curva:
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print("auc: ", auc)

# Curva ROC
from sklearn.metrics import roc_curve
plt.figure()
lw = 2
plt.plot(roc_curve(y_test, y_pred)[0], roc_curve(y_test, y_pred)[1], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# R Score (R^2 coefficient of determination)
from sklearn.metrics import r2_score
R = r2_score(y_test, y_pred)
print("R2: ", R)

# Guardar el modelo a un archivo
import joblib
joblib.dump(model, 'knn_model.pkl')
# Cargar el modelo desde el archivo
loaded_model = joblib.load('knn_model.pkl')
# Hacer predicciones con el modelo cargado
y_pred = model.predict(X_test)



# Decision Tree for breast cancer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import joblib

# Dividir en características (X) y objetivo (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para que todas las características tengan una escala similar
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea y entrena el modelo de árbol de decisión
model = DecisionTreeClassifier(max_depth=4, criterion='gini')
model.fit(X_train, y_train)

# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)

# Muestra el informe de evaluación del modelo entrenado
print(classification_report(y_test, y_pred))

# Matriz de confusión:
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cm)
# Gráfica de la matriz de confusión
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.show()

# Exactitud:
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# Sensibilidad:
recall = recall_score(y_test, y_pred)
print("Recall: ", recall)

# Precisión:
precision = precision_score(y_test, y_pred)
print("Precision: ", precision)

# Especificidad
specificity = recall_score(y_test, y_pred, pos_label=0)
print("Specificity: ", specificity)

# Puntuación F1:
f1 = f1_score(y_test, y_pred)
print("F1 score: ", f1)

# Área bajo la curva:
auc = roc_auc_score(y_test, y_pred)
print("AUC: ", auc)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# R Score (R^2 coefficient of determination)
R = r2_score(y_test, y_pred)
print("R2: ", R)

# Obtener la importancia de las características
feature_importances = model.feature_importances_
feature_names = X.columns

# Crear un DataFrame para visualizar la importancia de las características
feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Configurar el gráfico de barras
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='b')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance - Decision Tree')
plt.show()

# Visualizar el árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree")
plt.show()

# Guardar el modelo a un archivo
joblib.dump(model, 'decision_tree_model.pkl')

# Cargar el modelo desde el archivo
loaded_model = joblib.load('decision_tree_model.pkl')

# Hacer predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(X_test)
print("Predictions with loaded model:", y_pred_loaded)



# RNA for breast cancer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Dividir en características (X) y objetivo (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza los datos para que todas las características tengan una escala similar
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir la red neuronal con dropout
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Apaga aleatoriamente el 20% de las neuronas
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))  # Apaga aleatoriamente el 20% de las neuronas
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Configurar early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)


# Entrenar el modelo y almacenar el historial
history = model.fit(X_train, y_train, epochs=100, batch_size=10, 
                    validation_split=0.2, verbose=1,
          validation_data=(X_test, y_test), callbacks=[early_stopping])


# Evaluar el modelo
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Muestra el informe de evaluación del modelo entrenado
print(classification_report(y_test, y_pred))

# Matriz de confusión:
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cm)
# Gráfica de la matriz de confusión
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.show()

# Exactitud:
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# Sensibilidad:
recall = recall_score(y_test, y_pred)
print("Recall: ", recall)

# Precisión:
precision = precision_score(y_test, y_pred)
print("Precision: ", precision)

# Especificidad
specificity = recall_score(y_test, y_pred, pos_label=0)
print("Specificity: ", specificity)

# Puntuación F1:
f1 = f1_score(y_test, y_pred)
print("F1 score: ", f1)

# Área bajo la curva:
auc = roc_auc_score(y_test, y_pred)
print("AUC: ", auc)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Guardar el modelo a un archivo
model.save('neural_network_model_with_dropout.h5')

# Cargar el modelo desde el archivo
from tensorflow.keras.models import load_model
loaded_model = load_model('neural_network_model_with_dropout.h5')

# Hacer predicciones con el modelo cargado
y_pred_loaded = loaded_model.predict(X_test)
y_pred_loaded = (y_pred_loaded > 0.5).astype(int)
print("Predictions with loaded model:", y_pred_loaded)


# Learning Curves
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Obtener los SHAP values
explainer = shap.DeepExplainer(model, X_train)
# Obtener las explicaciones SHAP para el conjunto de prueba
shap_values = explainer.shap_values(X_test)
# Proporciona una visión general de la importancia de las características y su impacto en las predicciones.
shap.summary_plot(shap_values, X_test, feature_names=X.columns)


