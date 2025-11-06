import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga tu dataset (asegúrate de que la ruta sea correcta)
df = pd.read_csv('creditcard.csv')

# Vistazo rápido a las primeras filas
print(df.head())

# Revisa los tipos de datos y si hay valores nulos
print(df.info())

# Estadísticas descriptivas, especialmente de 'Amount' y 'Time'
print(df[['Time', 'Amount']].describe())