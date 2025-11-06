import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
df = pd.read_csv('creditcard.csv')

# Vistazo rápido a las primeras filas
print(df.head())

# Revisar los tipos de datos y si hay valores nulos
print(df.info())

# Estadísticas descriptivas, especialmente de 'Amount' y 'Time'
print(df[['Time', 'Amount']].describe())