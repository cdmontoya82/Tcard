import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix



#1 -----------Carga y Vistazo Inicial--------------
# Cargar dataset
df = pd.read_csv('/home/cristian/Documentos/MAchine_learning/creditcard.csv')

# Vistazo rápido a las primeras filas
print(df.head())

# Revisar los tipos de datos y si hay valores nulos
print(df.info())

# Estadísticas descriptivas, especialmente de 'Amount' y 'Time'
print(df[['Time', 'Amount']].describe())


#3.-----Análisis Exploratorio de Datos (EDA)-----------

# 1. ¡Verificar el desbalanceo!
print("Distribución de Clases:")
print(df['Class'].value_counts(normalize=True))

# 2. Visualizar el desbalanceo
sns.countplot(x='Class', data=df)
plt.title('Distribución de Transacciones (0: Genuina, 1: Fraude)')
plt.show()

# 3. Analizar la variable 'Amount'
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribución del Monto de la Transacción')
plt.xlim(0, 5000) # Limitar el eje X para ver mejor (hay valores atípicos muy altos)
plt.show()

# 4. Comparar 'Amount' en transacciones fraudulentas vs. genuinas
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Monto de Transacción vs. Clase')
plt.ylim(0, 200) # Limitar el eje Y para ver la mayoría de los casos
plt.show()

# guardar imagen

sns.countplot(x='Class', data=df)
plt.title('Distribución de Transacciones (0: Genuina, 1: Fraude)')
plt.savefig('grafico_distribucion.png') 
plt.show() # Esta línea ahora es opcional

#4. -------Preprocesamiento de Datos--------------

# Creamos un escalador
scaler = StandardScaler()

# Escalamos solo 'Amount' y 'Time'
df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Eliminamos las columnas originales
df = df.drop(['Time', 'Amount'], axis=1)

# Reordenamos las columnas (opcional, pero más limpio)
cols = ['scaled_Time', 'scaled_Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']
df = df[cols]

print(df.head())

#5.-------División de Datos (Train/Test Split)------




print("\n--- 5. Dividiendo los datos ---")

# Definimos X (features) e y (target)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividimos los datos (70% entrenamiento, 30% prueba)
# stratify=y es VITAL para asegurar que la proporción de fraude sea la misma en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)

print(f"Forma de X_train (antes de SMOTE): {X_train.shape}")
print(f"Forma de y_train (antes de SMOTE): {y_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
print(f"Forma de y_test: {y_test.shape}")


#6.------Balanceo de Datos de Entrenamiento (SMOTE)--------


print("\n--- 6. Aplicando SMOTE al set de entrenamiento ---")

# Inicializamos SMOTE
# random_state es para reproducibilidad
smote = SMOTE(random_state=42)

# Aplicamos SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Forma de X_train después de SMOTE: {X_train_resampled.shape}")
print("Distribución de clases en 'y_train' después de SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))



# --- 7. Entrenando el Modelo (Random Forest) ---
# from sklearn.linear_model import LogisticRegression # <-- Comenta o borra esta
from sklearn.ensemble import RandomForestClassifier # <-- Importa esta

print("\n--- 7. Entrenando el modelo (Random Forest) ---")

# 1. Inicializar el modelo
# n_estimators=100 es el número de "árboles" en el bosque.
# random_state=42 es para que el resultado sea reproducible.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Entrenar el modelo con los datos BALANCEADOS (¡los mismos de SMOTE!)
model.fit(X_train_resampled, y_train_resampled)

print("¡Modelo Random Forest entrenado exitosamente!")



#8.---Evaluación del Modelo----



print("\n--- 8. Evaluando el modelo en el set de prueba (Test) ---")

# 1. Hacer predicciones en el set de prueba (el que está desbalanceado)
y_pred = model.predict(X_test)

# 2. Generar el Reporte de Clasificación
# Class 0 = Genuina
# Class 1 = Fraude
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Genuina (0)', 'Fraude (1)']))

# 3. Generar la Matriz de Confusión
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 4. Visualizar la Matriz de Confusión (Opcional pero recomendado)
# Esto guardará la imagen como 'matriz_confusion.png'
print("\nGuardando gráfico de Matriz de Confusión...")
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred. Genuina', 'Pred. Fraude'], 
            yticklabels=['Real Genuina', 'Real Fraude'])
plt.title('Matriz de Confusión')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.savefig('matriz_confusion.png')

print("\n--- Proceso completado ---")