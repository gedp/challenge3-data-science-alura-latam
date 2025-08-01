# Telecom X – Parte 2: Predicción de Cancelación (Churn)

## 🧠 Objetivos del Desafío
* Preparar los datos para el modelado (tratamiento, codificación, normalización).
* Realizar análisis de correlación y selección de variables.
* Entrenar dos o más modelos de clasificación.
* Evaluar el rendimiento de los modelos con métricas.
* Interpretar los resultados, incluyendo la importancia de las variables.
* Crear una conclusión estratégica señalando los principales factores que influyen en la cancelación.

## 📚 Entregables

- Notebook con análisis completo
- Modelo final entrenado
- Informe de hallazgos



## 1. Preparación de Datos

### 1.1 Carga y exploración inicial de datos


```python
#importar el archivo de datos
import pandas as pd
#cargar el archivo de datos
data = pd.read_csv('https://raw.githubusercontent.com/anasayago/challenge2-data-science-latam/refs/heads/main/datos_tratados_2.csv',sep=',')
data.shape
```


```python
#primeras filas del archivo de datos
data.head()
```


```python
# Análisis inicial
data.info()
```


```python
#variables númericas
data.describe()
```


```python
print("\nDistribución de Abandono:")
print(data['Abandono'].value_counts(normalize=True))
```

 ⚠️ Comportamiento de la clase principal (Churn)
* No abandono:  73.463 %
* Abandono:  26.537 %


```python
#variables categóricas
data.describe(include='object')
```

### 1.2 Limpieza y Preparación de Datos


```python
# Eliminar columnas que no aportan información relevante
data = data.drop(columns=['ID Cliente','Costo Diario','Duración del Contrato (meses)'])
```

Las siguientes columnas no aportan información relevante
* Se elimina 'ID Cliente' por ser identificador único que no aporta información predictiva
* Se elimina 'Costo Diario' ya que deriva de 'Costo Mensual', lo que puede introducir multicolinealidad
* Se elimina 'Duración del Contrato (meses)' ya que existe Rango de Contrato y puede ser usado 


```python
#información general del conjunto de datos después de eliminar columnas
data.info()
```

### 1.3 Analisis de Gráficos para variables númericas relevantes

De acuerdo a la estadistica descriptiva las variables con variabilidad significativa y potencial predictivo son: 
- Costo Mensual
- Costo Total
- Cantidad de Servicios


```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,10))

columnas_numericas = [
    'Costo Total',
    'Costo Mensual',
    'Cantidad de Servicios Contratados'
]

for i, columna in enumerate(columnas_numericas, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Abandono', y=columna, data=data)
    plt.title(f'Boxplot de {columna} por Abandono')
    plt.xticks([0, 1], ['No Abandono', 'Abandono'])

plt.tight_layout()
plt.show()

```


```python
# distribución de las variables numéricas por Abandono(Crunch)
plt.figure(figsize=(16,12))

palette_colorblind = sns.color_palette("colorblind")[:2]

# Histogramas por estado de Abandono
for i, columna in enumerate(columnas_numericas, 1):
    plt.subplot(2, 2, i)

    sns.histplot(
        data=data,
        x=columna,
        hue='Abandono',
        multiple="stack",  
        palette=palette_colorblind,  
        bins=30  # Número de bins
    )

    plt.title(f'Distribución de {columna} por Abandono', fontsize=10)
    plt.xlabel(columna, fontsize=8)
    plt.ylabel('Frecuencia', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

plt.suptitle('Distribución de Variables Numéricas por Abandono', fontsize=14)
plt.tight_layout()
plt.show()

# Análisis estadístico complementario
for columna in columnas_numericas:
    print(f"\nEstadísticas para {columna}:")
    print(data.groupby('Abandono')[columna].describe())

```

⚠️ Las variables muestran las siguientes simetrías
* Costo Total
    - Fuerte asimetría positiva
    - Muchos valores bajos
    - Pocos valores extremadamente altos
* Costo Mensual
    - Moderadamente asimétrico
    - Sesgo positivo
    - Media ligeramente superior a mediana
* Cantidad de Servicios Contratados
    - Distribución cercana a normal
    - Rango discreto y limitado


```python
# Comportamiento de las variables binarias relevantes
variables_binarias = ['Tiene Pareja','Tiene Dependientes',
                      'Múltiples Líneas', 'Seguridad en Línea',
                      'Respaldo en Línea', 'Protección del Dispositivo']
for columna in variables_binarias:
    print(columna, data[columna].value_counts(normalize=True)*100)
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.color_palette("colorblind")

plt.figure(figsize=(16,12))
for i, variable in enumerate(variables_binarias, 1):
    plt.subplot(3, 2, i)
    # Gráfico de distribución con abandono
    data_plot = data.groupby([variable, 'Abandono']).size().unstack(fill_value=0)
    data_plot.plot(kind='bar', stacked=True, 
                   color=sns.color_palette("colorblind"), 
                   ax=plt.gca())
    
    plt.title(f'Distribución de {variable} por Abandono')
    plt.xlabel(variable)
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```


```python
# Chi-cuadrado para medir dependencia
from scipy.stats import chi2_contingency

for columna in variables_binarias:
    contingencia = pd.crosstab(data[columna], data['Abandono'])
    chi2, p_valor, _, _ = chi2_contingency(contingencia)
    print(f"{columna}: p-valor = {p_valor}")
```

De las variables binarias podemos decir: 
- Todas las variables tienen una relación estadísticamente significativa con el abandono
- Seguridad en Línea y Respaldo en Línea tienen la relación más fuerte
- Hay evidencia de que estas características influyen en la probabilidad de abandono


```python
# Analisis de variables binarias de servicios adicionales
servicios_internet = [
    'Seguridad en Línea', 
    'Respaldo en Línea', 
    'Protección del Dispositivo', 
    'Soporte Técnico', 
    'TV por Cable', 
    'Streaming de Películas'
]

# Chi-cuadrado para Múltiples Líneas
print("Dependencias con Múltiples Líneas:")
for servicio in servicios_internet:
    contingencia = pd.crosstab(data['Múltiples Líneas'], data[servicio])
    chi2, p_valor, _, _ = chi2_contingency(contingencia)
    print(f"{servicio}: p-valor = {p_valor}")

# Chi-cuadrado para Servicios de Internet
print("\nDependencias entre Servicios de Internet:")
for i in range(len(servicios_internet)):
    for j in range(i+1, len(servicios_internet)):
        servicio1 = servicios_internet[i]
        servicio2 = servicios_internet[j]
        contingencia = pd.crosstab(data[servicio1], data[servicio2])
        chi2, p_valor, _, _ = chi2_contingency(contingencia)
        print(f"{servicio1} vs {servicio2}: p-valor = {p_valor}")

```


```python
plt.figure(figsize=(15,10))
corr = data[servicios_internet].corr()
print("Mapa de calor de correlación entre Servicios de Internet:")
print(corr)

# Heatmap de correlación
plt.subplot(2,1,1)
sns.heatmap(
    data[servicios_internet].corr(), 
    annot=True, 
    cmap='coolwarm'
)
plt.title('Correlación entre Servicios de Internet')

# Distribución de servicios
plt.subplot(2,1,2)
servicios_count = data[servicios_internet].sum()
sns.barplot(x=servicios_count.index, y=servicios_count.values)
plt.title('Cantidad de Servicios Contratados')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

```


```python
def analisis_abandono_servicios(servicios):
    resultados = {}
    for servicio in servicios:
        abandono_con_servicio = data[data[servicio] == 1]['Abandono'].mean()
        abandono_sin_servicio = data[data[servicio] == 0]['Abandono'].mean()
        
        resultados[servicio] = {
            'Abandono con servicio': abandono_con_servicio,
            'Abandono sin servicio': abandono_sin_servicio,
            'Diferencia': abandono_con_servicio - abandono_sin_servicio
        }
    
    return pd.DataFrame.from_dict(resultados, orient='index')

# Análisis de abandono
print("Análisis de Abandono por Servicios:")
print(analisis_abandono_servicios(servicios_internet))

```


```python
# Número de servicios contratados
data['Servicios_Contratados'] = data[servicios_internet].sum(axis=1)

plt.figure(figsize=(10,6))
sns.boxplot(x='Servicios_Contratados', y='Costo Mensual', hue='Abandono', data=data)
plt.title('Servicios Contratados vs Costo Mensual y Abandono')
plt.show()

```

Analisis de Servicios adicionales de Servicio Telefónico e Internet
- Todos los servicios reducen la probabilidad de abandono
- Priorizar Seguridad en Línea y Soporte Técnico
- Servicios con alta correlación pueden ser empaquetados

### 1.4 Gráficos de variables categóricas


```python
# Variables categóricas
categoricas = ['Género', 'Servicio de Internet', 'Tipo de Contrato', 'Método de Pago', 'Rango de Contrato']
plt.figure(figsize=(16,20))

for i, variable in enumerate(categoricas, 1):
    plt.subplot(3, 2, i)

    # Gráfico de distribución con abandono
    data_plot = data.groupby([variable, 'Abandono']).size().unstack(fill_value=0)
    data_plot_percent = data_plot.div(data_plot.sum(axis=1), axis=0) * 100

    data_plot_percent.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title(f'Distribución de Abandono por {variable}')
    plt.xlabel(variable)
    plt.ylabel('Porcentaje')
    plt.legend(title='Abandono', labels=['No Abandono', 'Abandono'])
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

```


```python
# Chi-cuadrado para evaluar dependencia
from scipy.stats import chi2_contingency

for variable in categoricas:
    contingencia = pd.crosstab(data[variable], data['Abandono'])
    chi2, p_value, dof, expected = chi2_contingency(contingencia)

    print(f"\nVariable: {variable}")
    print(f"Chi-cuadrado: {chi2}")
    print(f"P-value: {p_value}")
    print("Significativa:" + " Sí" if p_value < 0.05 else " No")

```

### 1.5 Información Mutua con Variable Abandono


```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_encoded = data[categoricas].apply(le.fit_transform)

# Calcular información mutua
mi_scores = mutual_info_classif(X_encoded, data['Abandono'])

# Visualizar
plt.figure(figsize=(10,6))
plt.bar(categoricas, mi_scores)
plt.title('Información Mutua con Variable Abandono')
plt.xlabel('Variables Categóricas')
plt.ylabel('Puntuación de Información Mutua')
plt.show()

# Imprimir scores
for var, score in zip(categoricas, mi_scores):
    print(f"{var}: {score}")

```


```python
# Análisis de la tasa de abandono por rango de contrato

plt.figure(figsize=(10,6))
churn_by_contract = data.groupby('Rango de Contrato')['Abandono'].mean()
churn_by_contract.plot(kind='bar')
plt.title('Tasa de Abandono por Rango de Contrato')
plt.ylabel('Tasa de Abandono')
plt.xlabel('Rango de Contrato')
plt.tight_layout()
plt.show()
```

### 1.6 Detección y tratamiento de outliers



```python
def agrupar_por_churn(rango):
    if rango in ['0-10', '11-20']:
        return 'Riesgo Alto'
    elif rango in ['21-30', '31-40']:
        return 'Riesgo Medio'
    else:
        return 'Riesgo Bajo'

data['Rango_Contrato'] = data['Rango de Contrato'].apply(agrupar_por_churn)
#Eliminar columna original
data = data.drop(columns=['Rango de Contrato'])
```

⚠️ Se aplicaran transformaciones logarítmica, boxcx y raiz cuadrada


```python
import numpy as np
from scipy import stats
# Transformaciones
data['log_costo_total'] = np.log1p(data['Costo Total'])
data['sqr_costo_total']= np.sqrt(data['Costo Total'])
#data['log_costo_total'] = stats.boxcox(data['Costo Total'] + 1)
data['boxcx_costo_mensual'], _ = stats.boxcox(data['Costo Mensual'] + 1)
data['sqr_costo_mensual'] = np.sqrt(data['Costo Mensual'])
#data['sqr_duracion_contrato'] = np.sqrt(data['Duración del Contrato (meses)'])
data['sqr_servicios'] = np.sqrt(data['Cantidad de Servicios Contratados'])
data.head()
```


```python
plt.figure(figsize=(15,10))

# Columnas numéricas
columnas_numericas = [
    'log_costo_total',
    'boxcx_costo_mensual',
    'Cantidad de Servicios Contratados',
    'sqr_costo_total',
    'sqr_costo_mensual',
    'sqr_servicios'
]

# Visualización de boxplots por estado de Abandono de las columnas numéricas
plt.figure(figsize=(16,12))
for i, columna in enumerate(columnas_numericas, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='Abandono', y=columna, data=data)
    plt.title(f'Boxplot de {columna} por Abandono')
    plt.xticks([0, 1], ['No Abandono', 'Abandono'])

plt.tight_layout()
plt.show()

```


```python
#eliminar variables transformadas porque no logran una mejora significativa en la distribución
data = data.drop(columns=['log_costo_total', 'boxcx_costo_mensual'])
```


```python
#eliminar las variables transformadas, posteriormene se analizará si se vuelven a incluir
data = data.drop(columns=['Costo Total','Costo Mensual','Cantidad de Servicios Contratados'])
```


```python
data.columns
```

### 1.7 Codificación de variables categóricas


```python
# One-Hot Encoding con get_dummies
categoricas = ['Género', 'Tipo de Contrato', 'Método de Pago','Rango_Contrato', 'Servicio de Internet']

# Aplicación básica
data_encoded = pd.get_dummies(data, columns=categoricas)

# Opciones adicionales
data_encoded = pd.get_dummies(
    data,
    columns=categoricas,
    prefix=categoricas,  # Prefijo para nuevas columnas
    drop_first=True     # Elimina primera categoría para evitar multicolinealidad
)
print("\nColumnas después de One-Hot Encoding:")
print(data_encoded.shape)
print(data_encoded.columns)
```


```python
# analisis de correlación
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16,12))
correlation_matrix = data_encoded.corr()

# Extraer variables de interés
print("\nVariables de Interes en Matriz de Correlación:")
print(correlation_matrix['Abandono'].sort_values(ascending=False))

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación')
plt.show()
```

* Correlaciones Positivas (Aumentan probabilidad de Abandono):
    - Tipo de Contrato_Mensual (0.405)
    - Servicio de Internet_Fibra Óptica (0.308)
    - sqr_costo_mensual (0.203)
    - Facturación Sin Papel (0.192)
    - TV por Cable (0.165)
    - Streaming de Películas (0.163)
    - Mayor de 65 años (0.151)

* Correlaciones Negativas (Disminuyen probabilidad de Abandono):
    - Rango_Contrato_Riesgo Bajo (-0.284)
    - Tipo de Contrato_Bianual (-0.302)
    - Servicio de Internet_No internet service (-0.228)
    - sqr_costo_total (-0.223)
    - Tiene Dependientes (-0.164)
    - Tiene Pareja (-0.150)
    - Método de Pago_Tarjeta de crédito (automático) (-0.134)

## 2. Preparación para Modelado

### 2.1 Separación de Datos


```python
# Variable dependiente
y = data_encoded['Abandono']
# Variables independientes
X = data_encoded.drop(columns=['Abandono'])
```


```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### 2.2 Balanceo de clases


```python
# Contar distribución de clases
distribucion = data_encoded['Abandono'].value_counts(normalize=True)
print(distribucion)

# Calcular ratio de desbalanceo
ratio = len(data_encoded[data_encoded['Abandono']==0]) / len(data[data['Abandono']==1])
print(f"Ratio de desbalanceo: {ratio}")
```

**No se aplicará balanceo** ya que tiene una **ratio de desbalanceo moderado** de 2.77 (no extremo), por lo tanto no es un desbalanceo crítico (generalmente se considera crítico >5)

## 3. Entrenamiento de Modelos

### 3.1 Selección de los Modelos
- Random Forest: ✅ Adecuado 
    - Robusto con distintos tipos de variables
    - Buena interpretabilidad
    - Captura no linealidades
≈
- CatBoost: ✅ Muy bueno
    - Excelente con variables categóricas
    - Maneja binarias perfectamente
    - Reduce sesgo de codificación

- Random Forest: ✅ Adecuado 
    - Robusto con distintos tipos de variables
    - Buena interpretabilidad
    - Captura no linealidades

### 3.2 Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

# Crear y entrenar el modelo Random Forest
model_rf = RandomForestClassifier(n_estimators=100, 
                                  random_state=42)
model_rf.fit(X_train, y_train)

# Hacer predicciones
y_pred_rf = model_rf.predict(X_test)
```

### 3.3 Catboost


```python
%pip install catboost
```


```python
from catboost import CatBoostClassifier

# Configuración del modelo CatBoost
model_cb = CatBoostClassifier(
    iterations=300,  # Aumentar iteraciones
    learning_rate= 0.1,  # Probar diferentes tasas
    depth=6,  # Variar profundidad
    l2_leaf_reg=3,  # Regularización
    random_seed=42,
    verbose=0  # Quita mensajes de progreso
)

# Entrenar el modelo CatBoost
model_cb.fit(X_train, y_train)
# Hacer predicciones
y_pred_cb = model_cb.predict(X_test)

```

### 3.4 Xgboost


```python
%pip install xgboost
```


```python
import xgboost as xgb

# Entrenar el modelo XGBoost
model_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model_xgb.fit(X_train, y_train)
# Hacer predicciones
y_pred_xgb = model_xgb.predict(X_test)
```

## 4. Evaluación de los modelos

### 4.1 Métricas de Random Forest


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Random Forest
print("Métricas Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

```

### 4.2 Métricas de CatBoost


```python
# CatBoost
print("\nMétricas de CatBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_cb))
print("Precision:", precision_score(y_test, y_pred_cb))
print("Recall:", recall_score(y_test, y_pred_cb))
print("F1-score:", f1_score(y_test, y_pred_cb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_cb))
print(classification_report(y_test, y_pred_cb))
```

### 4.3 Métricas de XGBoost


```python
# XGBoost
print("\nMetricas de XGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1-score:", f1_score(y_test, y_pred_xgb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
```

## 5 Ajuste de modelos
#### 5.1 Random Forest


```python
# Aumentar la complejidad del modelo
model_ajustado_rf = RandomForestClassifier(
    n_estimators=150,        # Más árboles
    max_depth=10,            # Mayor profundidad máxima
    min_samples_split=5,     # Menos muestras para dividir un nodo
    min_samples_leaf=3,      # Menos muestras en una hoja
    max_features='sqrt',     # Número de características a considerar en cada división
    random_state=42
)

model_ajustado_rf.fit(X_train, y_train)
y_pred_ajustado_rf= model_ajustado_rf.predict(X_test)
```

#### 5.2 CatBoost


```python
# Aumentar la complejidad y ajustar parámetros de CatBoost
from catboost import CatBoostClassifier
model_ajustado_cb = CatBoostClassifier(
    iterations=500,          # Más iteraciones
    learning_rate=0.01,     # Tasa de aprendizaje más baja
    depth=8,                # Mayor profundidad
    l2_leaf_reg=3,          # Regularización L2
    random_seed=42,
    verbose=0               # Quitar mensajes de progreso
)
model_ajustado_cb.fit(X_train, y_train)
y_pred_ajustado_cb = model_ajustado_cb.predict(X_test)
```

#### 5.3 XGBoost


```python
# Aumentar la complejidad y ajustar parámetros de XGBoost
model_ajustado_xgb = xgb.XGBClassifier(
    n_estimators=150,         # Más árboles
    max_depth=10,             # Mayor profundidad máxima
    learning_rate=0.05,       # Tasa de aprendizaje más baja
    subsample=0.8,            # Porcentaje de muestras para cada árbol
    colsample_bytree=0.8,     # Porcentaje de columnas para cada árbol
    eval_metric='logloss',
    random_state=42,
    reg_alpha=0.1,          # Regularización L1
    reg_lambda=1.0          # Regularización L2
)

model_ajustado_xgb.fit(X_train, y_train)
y_pred_ajustado_xgb = model_ajustado_xgb.predict(X_test)
```

### 5.4 Evaluación de modelos ajustados



```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# Random Forest
print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_ajustado_rf))
print("Precision:", precision_score(y_test, y_pred_ajustado_rf))
print("Recall:", recall_score(y_test, y_pred_ajustado_rf))
print("F1-score:", f1_score(y_test, y_pred_ajustado_rf))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_ajustado_rf))
print(classification_report(y_test, y_pred_ajustado_rf))
```


```python
# CatBoost
print("\nCatBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_ajustado_cb))
print("Precision:", precision_score(y_test, y_pred_ajustado_cb))
print("Recall:", recall_score(y_test, y_pred_ajustado_cb))
print("F1-score:", f1_score(y_test, y_pred_ajustado_cb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_ajustado_cb))
print(classification_report(y_test, y_pred_ajustado_cb))
```


```python
# XGBoost
print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_ajustado_xgb))
print("Precision:", precision_score(y_test, y_pred_ajustado_xgb))
print("Recall:", recall_score(y_test, y_pred_ajustado_xgb))
print("F1-score:", f1_score(y_test, y_pred_ajustado_xgb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_ajustado_xgb))
print(classification_report(y_test, y_pred_ajustado_xgb))
```

### 5.5 Entrenamiento y evaluación en datos de entrenamiento


```python
# Entrenamiento y evaluación en datos de entrenamiento
y_train_pred_ajustado_rf = model_ajustado_rf.predict(X_train)
y_train_pred_ajustado_cb = model_ajustado_cb.predict(X_train)
y_train_pred_xgb_ajustado = model_ajustado_xgb.predict(X_train)

print("Random Forest - Entrenamiento:", accuracy_score(y_train, y_train_pred_ajustado_rf))
print("Random Forest - Prueba:", accuracy_score(y_test, y_pred_ajustado_rf))

print("CatBoost - Entrenamiento:", accuracy_score(y_train, y_train_pred_ajustado_cb))
print("CatBoost - Prueba:", accuracy_score(y_test, y_pred_ajustado_cb))

print("XGBoost - Entrenamiento:", accuracy_score(y_train, y_train_pred_xgb_ajustado))
print("XGBoost - Prueba:", accuracy_score(y_test, y_pred_ajustado_xgb))
```

Análisis de Overfitting:

- Random Forest
    - Entrenamiento: 85.00%
    - Prueba: 79.42%
    - Diferencia: 5.58% (Moderado overfitting)

- CatBoost
    - Entrenamiento: 84.47%
    - Prueba: 79.35%
    - Diferencia: 5.12% (Moderado overfitting)

- XGBoost
    - Entrenamiento: 94.50%
    - Prueba: 77.86%
    - Diferencia: 16.64% (Alto overfitting)

**Interpretación**
- XGBoost: Mayor sobreajuste
- Random Forest: Mejor generalización
- CatBoost: Equilibrio intermedio

### 5.6 Validación cruzada


```python
# Validación cruzada para evaluar la estabilidad del modelo
from sklearn.model_selection import cross_val_score

rf_scores = cross_val_score(model_ajustado_rf, X, y, cv=5, scoring='accuracy')
print("Random Forest - Validación cruzada (accuracy):", rf_scores.mean())

cb_scores = cross_val_score(model_ajustado_cb, X, y, cv=5, scoring='accuracy')
print("CatBoost - Validación cruzada (accuracy):", cb_scores.mean())

xgb_scores = cross_val_score(model_ajustado_xgb, X, y, cv=5, scoring='accuracy')
print("XGBoost - Validación cruzada (accuracy):", xgb_scores.mean())
```

### 5.7 Dataframe de variables reducidas

Se considera reducir el dataset para disminuye la complejidad del modelo, reducir el ruido y sobretodo manejar el sobreajuste


```python
#importancia de las características
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(model_rf.feature_importances_)
}).sort_values('importance', ascending=False)
print(importance.head())
```


```python
features = X.columns
importances = model_rf.feature_importances_
#crear DataFrame X_reduced
low_importance = [feature for feature, importance in zip(features, importances) if importance < 0.01]
print("Variables a eliminar:", low_importance)
X_reduced = X.drop(columns=low_importance)
```

#### 5.7.1 Random Forest con variables reducidas


```python
#Modelo Random Forest con las variables reducidas
rdmforest_model_reducido = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42
)

rdmforest_model_reducido.fit(X_reduced, y)
y_pred_rf_reducido = rdmforest_model_reducido.predict(X_test.drop(columns=low_importance))
```

#### 5.7.2 CatBoost con variables reducidas


```python
#Modelo CatBoost con las variables reducidas
model_ajustado_cb_reducido = CatBoostClassifier(
    iterations=500,
    learning_rate=0.01,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)

model_ajustado_cb_reducido.fit(X_reduced, y)
y_pred_cb_reducido = model_ajustado_cb_reducido.predict(X_test.drop(columns=low_importance))
```

#### 5.7.3 XGBoost con variables reducidas


```python
#Modelo XGBoost con las variables reducidas
model_ajustado_xgb_reducido = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    reg_alpha=0.1,
    reg_lambda=1.0
)
model_ajustado_xgb_reducido.fit(X_reduced, y)
y_pred_xgb_reducido = model_ajustado_xgb_reducido.predict(X_test.drop(columns=low_importance))
```

#### 5.7.4 Evaluación de Modelos con variables reducidas


```python
#Comparar resultados de los modelos con variables reducidas
print("Random Forest con variables reducidas:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_reducido))
print("Precision:", precision_score(y_test, y_pred_rf_reducido))
print("Recall:", recall_score(y_test, y_pred_rf_reducido))
print("F1-score:", f1_score(y_test, y_pred_rf_reducido))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_rf_reducido))
print(classification_report(y_test, y_pred_rf_reducido))
```


```python
#Comparar resultados de los modelos con variables reducidas
print("CatBoost con variables reducidas:")
print("Accuracy:", accuracy_score(y_test, y_pred_cb_reducido))
print("Precision:", precision_score(y_test, y_pred_cb_reducido))
print("Recall:", recall_score(y_test, y_pred_cb_reducido))
print("F1-score:", f1_score(y_test, y_pred_cb_reducido))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_cb_reducido))
print(classification_report(y_test, y_pred_cb_reducido))

```


```python
#Comparar resultados de los modelos con variables reducidas
print("XGBoost con variables reducidas:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb_reducido))
print("Precision:", precision_score(y_test, y_pred_xgb_reducido))
print("Recall:", recall_score(y_test, y_pred_xgb_reducido))
print("F1-score:", f1_score(y_test, y_pred_xgb_reducido))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_xgb_reducido))
print(classification_report(y_test, y_pred_xgb_reducido))
```

Observamos las siguientes mejoras: 
- Random Forest
    - Antes: Accuracy 0.7942
    - Después: Accuracy 0.8318
    - Mejora: +0.0376 (+4.74%)
- CatBoost
    - Después: Accuracy 0.8212
    - Antes: Accuracy 0.7935
    - Mejora: +0.0277 (+3.49%)
- XGBoost
    - Antes: Accuracy 0.7786
    - Después: Accuracy 0.9383
    - Mejora: +0.1597 (+20.52%)

**XGBoost** muestra la mejora significativa y se observa que la reducción de variables REDUJO el sobreajuste. Tambien observamos que todos los modelos mejoraron su rendimiento.

Se refuerza la idea que **XGBoost** con **variables reducidas** es el mejor modelo.

#### 5.7.5 Validación Cruzada con variables reducidas


```python
# Validación cruzada para evaluar la estabilidad del modelo con variables reducidas
rf_reduced_scores = cross_val_score(rdmforest_model_reducido, X_reduced, y, cv=5, scoring='accuracy')
print("Random Forest Reducido - Validación cruzada (accuracy):", rf_reduced_scores.mean())
print("CatBoost Reducido - Validación cruzada (accuracy):", cross_val_score(model_ajustado_cb_reducido, X_reduced, y, cv=5, scoring='accuracy').mean())
print("XGBoost Reducido - Validación cruzada (accuracy):", cross_val_score(model_ajustado_xgb_reducido, X_reduced, y, cv=5, scoring='accuracy').mean())
```

Justificación para mantener XGBoost:
A pesar de tener una pequeña variación con los otros modelos en la validación cruzada, podemo decir que el rendimiento es mejor: 
- Accuracy en prueba: 0.9383
- Precision: 0.9042
- Recall: 0.8583
- F1-score: 0.8807
Esta diferencia es estadísticamente aceptable y no compromete la generalización del modelo

**XGBoost** tambien técnicamente tiene un manejo eficiente de variables, capacidad de aprendizaje incremental y tratamiento robusto de datos no lineales.

### 5.8 Ajuste de Umbral

Realizar análisis de curva ROC para encontrar umbral óptimo que maximice rendimiento del modelo.


```python
# Probabilidades de cada modelo sin cambios
from sklearn.metrics import roc_curve, roc_auc_score

y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
y_prob_cb = model_cb.predict_proba(X_test)[:, 1]
y_prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

#Curva ROC y AUC para Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
# Curva ROC y AUC para CatBoost
fpr_cb, tpr_cb, _ = roc_curve(y_test, y_prob_cb)
auc_cb = roc_auc_score(y_test, y_prob_cb)
# Curva ROC y AUC para XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Graficar todas las curvas ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.2f})')
plt.plot(fpr_cb, tpr_cb, label=f'LightGBM (AUC={auc_cb:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Comparación de Curvas ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.tight_layout()
plt.show()

print(f"AUC Random Forest: {auc_rf:.2f}")
print(f"AUC XGBoost: {auc_xgb:.2f}")
print(f"AUC LightGBM: {auc_cb:.2f}")
```


```python
# Probabilidades de cada modelo ajustado
y_prob_rf = model_ajustado_rf.predict_proba(X_test)[:, 1]
y_prob_cb = model_ajustado_cb.predict_proba(X_test)[:, 1]
y_prob_xgb = model_ajustado_xgb.predict_proba(X_test)[:, 1]

#Curva ROC y AUC para Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
# Curva ROC y AUC para CatBoost
fpr_cb, tpr_cb, _ = roc_curve(y_test, y_prob_cb)
auc_cb = roc_auc_score(y_test, y_prob_cb)
# Curva ROC y AUC para XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Graficar todas las curvas ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.2f})')
plt.plot(fpr_cb, tpr_cb, label=f'LightGBM (AUC={auc_cb:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Comparación de Curvas ROC de Modelos Ajustados')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.tight_layout()
plt.show()

print(f"AUC Random Forest ajustado: {auc_rf:.2f}")
print(f"AUC XGBoost ajustado: {auc_xgb:.2f}")
print(f"AUC LightGBM ajustado: {auc_cb:.2f}")
```


```python
# Probabilidades de cada modelo reducido


y_prob_rf = rdmforest_model_reducido.predict_proba(X_test.drop(columns=low_importance))[:, 1]
y_prob_cb = model_ajustado_cb_reducido.predict_proba(X_test.drop(columns=low_importance))[:, 1]
y_prob_xgb = model_ajustado_xgb_reducido.predict_proba(X_test.drop(columns=low_importance))[:, 1]
#Curva ROC y AUC para Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
# Curva ROC y AUC para CatBoost
fpr_cb, tpr_cb, _ = roc_curve(y_test, y_prob_cb)
auc_cb = roc_auc_score(y_test, y_prob_cb)
# Curva ROC y AUC para XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# Graficar todas las curvas ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.2f})')
plt.plot(fpr_cb, tpr_cb, label=f'LightGBM (AUC={auc_cb:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Comparación de Curvas ROC de Modelos con Variables Reducidas')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.tight_layout()
plt.show()

print(f"AUC Random Forest Reducido: {auc_rf:.2f}")
print(f"AUC XGBoost Reducido: {auc_xgb:.2f}")
print(f"AUC LightGBM Reducido: {auc_cb:.2f}")
```


```python
#Ajuste de umbral para mejorar precisión y recall
y_proba = rdmforest_model_reducido.predict_proba(X_test.drop(columns=low_importance))[:, 1]

umbral = 0.45
y_pred_rf_umbral = (y_proba > umbral).astype(int)

print("Random Forest con umbral ajustado (0.45):")
print(classification_report(y_test, y_pred_rf_umbral))
```


```python
# Obtener probabilidades de abandono con CatBoost
y_proba_cb = model_ajustado_cb_reducido.predict_proba(X_test.drop(columns=low_importance))[:, 1]
umbral = 0.4
y_pred_cb_umbral = (y_proba_cb > umbral).astype(int)
print("CatBoost con umbral ajustado (0.4):")
print(classification_report(y_test, y_pred_cb_umbral))
```


```python
#Obtener probabilidades de abandono con XGBoost
y_proba_xgb = model_ajustado_xgb_reducido.predict_proba(X_test.drop(columns=low_importance))[:, 1]
umbral = 0.45
y_pred_xgb_umbral = (y_proba_xgb > umbral).astype(int)
print("XGBoost con umbral ajustado (0.4):")
print(classification_report(y_test, y_pred_xgb_umbral))
```

🔍 Análisis de Resultados
- XGBoost se destaca con la mejor precisión (0.96) y recall (0.95), lo que indica que es muy efectivo en la identificación de la clase positiva (1) y tiene un bajo número de falsos positivos.
- Random Forest tiene un buen rendimiento, especialmente en recall (0.90), lo que sugiere que es efectivo en la detección de positivos, aunque su precisión es un poco menor en comparación con XGBoost.
- CatBoost muestra un rendimiento sólido, pero no alcanza los niveles de XGBoost. Su precisión es alta, pero el recall es un poco más bajo, lo que puede indicar que está perdiendo algunos verdaderos positivos.

## 6. Interpretación
### 6.1 Analisis de las variables relevantes de XGBoost Reducido

Se revisan las variables mas relevantes para la predicción de cancelación en el modelo XGBoost reducido (el cual ha sido el modelo con mejores resultados ) y Random Forest reducido, este enfoque complementado permite identificar los factores críticos que influyen en la deserción de clientes que permitan desarrollar estrategias de retención más efectivas y personalizadas. Con esta técnica se busca equilibrar la complejidad computacional, interpretabilidad y poder predictivo.


```python
importancia = model_ajustado_xgb_reducido.feature_importances_
feature_importance = sorted(
    zip(X.columns, importancia),
    key=lambda x: x[1],
    reverse=True
)
importance_xgb = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
print("\nImportancia de las características en XGBoost con variables reducidas:")
print(importance_xgb.head(10))

```


```python
gain_xgb = model_ajustado_xgb_reducido.get_booster().get_score(importance_type='gain')
gain_xgb = sorted(gain_xgb.items(), key=lambda x: x[1], reverse=True)
importance_gain_xgb = pd.DataFrame(gain_xgb, columns=['Feature', 'Importance'])
print("\nImportancia de Características por Gain:")
print(importance_gain_xgb.head(10))

```


```python
#Cobertura para modelo XGBoost
cover_xgb = model_ajustado_xgb_reducido.get_booster().get_score(importance_type='cover')
cover_xgb = sorted(cover_xgb.items(), key=lambda x: x[1], reverse=True)
importance_cover_xgb = pd.DataFrame(cover_xgb, columns=['Feature', 'Importance'])
print("\nImportancia de Características por Cobertura:")
print(importance_cover_xgb.head(10))

```


```python
# Variables top consistentes aparece en top 5 para Importance, Gain y Cover
top_features = set(importance_xgb['Feature'].head(5)) & set(importance_gain_xgb['Feature'].head(5)) & set(importance_cover_xgb['Feature'].head(5))
print("\nVariables Top Consistentes en Importancia, Gain y Cover:")
print(top_features)
```

### 6.2 Analisis complementario de variables relevantes Random Forest


```python
from sklearn.inspection import permutation_importance
# Importancia nativa del modelo
importancia_nativa = rdmforest_model_reducido.feature_importances_
#imprimir 10 de las variables más importantes de la importancia_nativa
features = X_reduced.columns
importances = importancia_nativa
feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
print("\nImportancia Nativa del Modelo:")
print(importance_df.head(10))

# Cálculo de reducción de impureza
importancia_impureza = rdmforest_model_reducido.feature_importances_
# permutacion_importance para evaluar la importancia de las características
#Imprimir 10 de las variables más importantes de la importancia_impureza
features = X_reduced.columns
importances = importancia_impureza
feature_importance_impureza = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
importance_df_impureza = pd.DataFrame(feature_importance_impureza, columns=['Feature', 'Importance'])
print("\nImportancia por Reducción de Impureza:")
print(importance_df_impureza.head(10))

# Evaluar la importancia de las características con permutación
rdmforest_model_reducido.fit(X_reduced, y)
result = permutation_importance(
    rdmforest_model_reducido,
    X_test.drop(columns=low_importance),
    y_test,
    n_repeats=30,
    random_state=42
)
importancia_permutacion = result.importances_mean
#Imprimir 10 de las variables más importantes de la importancia_permutacion
features = X_reduced.columns
importances = importancia_permutacion
feature_importance_permutacion = sorted(zip(features, importances), key=lambda x: x [1], reverse=True)
importance_df_permutacion = pd.DataFrame(feature_importance_permutacion, columns=['Feature', 'Importance'])
print("\nImportancia por Permutación:")
print(importance_df_permutacion.head(10))
```


```python
#Graficar las importancias
plt.figure(figsize=(10, 6))
plt.barh(X_reduced.columns, importancia_nativa, label='Importancia Nativa', color='yellow', alpha=0.6)
plt.barh(X_reduced.columns, importancia_impureza, label='Reducción de Impureza', color='red', alpha=0.6)
plt.barh(X_reduced.columns, importancia_permutacion, label='Importancia por Permutación', color='green', alpha=0.6)
plt.xlabel('Importancia')
plt.title('Comparación de Importancias de Características')
plt.legend()
plt.tight_layout()
```


```python
# Variables top consistentes para random forest
top_features_rf = set(X_reduced.columns[importancia_nativa > 0.01]) & set(X_reduced.columns[importancia_impureza > 0.01]) & set(X_reduced.columns[importancia_permutacion > 0.01])
print("\nVariables Top Consistentes en Importancias de Random Forest:")
print(top_features_rf)

```

## 7. Anexos


```python
#Generación del Modelo XGBoost con las variables reducidas entrenado
# Guardar el modelo entrenado
import joblib
joblib.dump(model_ajustado_xgb_reducido, 'modelo_xgboost_reducido.pkl')
```


```python
%pip install nbconvert
```


```python
import nbconvert
import nbformat

# Convertir notebook a Markdown
exporter = nbconvert.MarkdownExporter()
markdown_output, _ = exporter.from_filename('TelecomX_LATAM.ipynb')


# Guardar el markdown
with open('TelecomX_LATAM.md', 'w') as f:
    f.write(markdown_output)
```
