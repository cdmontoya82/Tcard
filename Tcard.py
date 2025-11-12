import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


#1 -----------Carga y Vistazo Inicial--------------
# Cargar dataset
df = pd.read_csv('/home/cristian/Documentos/MAchine_learning/creditcard.csv')

# Vistazo rápido a las primeras filas
print(df.head())

# Revisar los tipos de datos y si hay valores nulos
print(df.info())

# Estadísticas descriptivas, especialmente de 'Amount' y 'Time'
print(df[['Time', 'Amount']].describe())


#-----Análisis Exploratorio de Datos (EDA)-----------

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

# -------Preprocesamiento de Datos--------------

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

#-------División de Datos (Train/Test Split)------




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


#------Balanceo de Datos de Entrenamiento (SMOTE)--------


print("\n--- 6. Aplicando SMOTE al set de entrenamiento ---")

# Inicializamos SMOTE
# random_state es para reproducibilidad
smote = SMOTE(random_state=42)

# Aplicamos SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Forma de X_train después de SMOTE: {X_train_resampled.shape}")
print("Distribución de clases en 'y_train' después de SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

#----------Entrenamiento del Modelo (Línea Base)------------



print("\n--- 7. Entrenando el modelo (Regresión Logística) ---")

# 1. Inicializar el modelo
# max_iter se aumenta por si la convergencia tarda
model = LogisticRegression(random_state=42, max_iter=1000)

# 2. Entrenar el modelo con los datos BALANCEADOS
model.fit(X_train_resampled, y_train_resampled)

print("¡Modelo entrenado exitosamente!")