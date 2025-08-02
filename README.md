# Challenge Telecom X – Parte 2: Predicción de Cancelación (Churn)

## 🎯Descripción del Proyecto

Desarrollar un modelo predictivo para identificar clientes con alto riesgo de cancelación de servicios, permitiendo a la empresa tomar acciones preventivas y estratégicas para reducir la tasa de abandono y optimizar la retención de clientes.


## 📊 Objetivos del Desafío
* Preparar los datos para el modelado (tratamiento, codificación, normalización).

* Realizar análisis de correlación y selección de variables.

* Entrenar dos o más modelos de clasificación.

* Evaluar el rendimiento de los modelos con métricas.

* Interpretar los resultados, incluyendo la importancia de las variables.

* Crear una conclusión estratégica señalando los principales factores que influyen en la cancelación.

### Componentes principales
* Análisis exploratorio de datos
* Preprocesamiento e ingenieria de caracteristicas
* Modelado predictivo de clasificación
* Evaluación de factores de riesgo de cancelación

## Estructura del proyecto

- **TelecomX_LATAM.ipynb**: Notebook principal con todo el proceso de análisis, desde la carga y limpieza de datos hasta la visualización y conclusiones.
- **datos_tratados**: Fuente de datos tratados en proyecto anterior. 
- **modelo_xgboost_reducido.plk**: Modelo XGBOOST generado
- **Informe_TelecomX_Cancelacion**: Informe detallado sobre resultados y hallazgos

### Datos iniciales 
```
Data columns (total 24 columns):
 #   Column                             Non-Null Count  Dtype  
---  ------                             --------------  -----  
 0   ID Cliente                         7043 non-null   object 
 1   Abandono                           7043 non-null   int64  
 2   Género                             7043 non-null   object 
 3   Mayor de 65 años                   7043 non-null   bool   
 4   Tiene Pareja                       7043 non-null   int64  
 5   Tiene Dependientes                 7043 non-null   int64  
 6   Duración del Contrato (meses)      7043 non-null   int64  
 7   Servicio Telefónico                7043 non-null   int64  
 8   Múltiples Líneas                   7043 non-null   int64  
 9   Servicio de Internet               7043 non-null   object 
 10  Seguridad en Línea                 7043 non-null   int64  
 11  Respaldo en Línea                  7043 non-null   int64  
 12  Protección del Dispositivo         7043 non-null   int64  
 13  Soporte Técnico                    7043 non-null   int64  
 14  TV por Cable                       7043 non-null   int64  
 15  Streaming de Películas             7043 non-null   int64  
 16  Tipo de Contrato                   7043 non-null   object 
 17  Facturación Sin Papel              7043 non-null   int64  
 18  Método de Pago                     7043 non-null   object 
 19  Costo Mensual                      7043 non-null   float64
 20  Costo Total                        7043 non-null   float64
 21  Costo Diario                       7043 non-null   float64
 22  Rango de Contrato                  7043 non-null   object 
 23  Cantidad de Servicios Contratados  7043 non-null   int64  
```

## Estructura del Notebook TelecomX_LATAM.ipynb

El notebook cubre la fase de análisis y modelado de datos, desde la preparación inicial hasta la interpretación de los resultados finales. 
• 1. Preparación de Datos
    ◦ 1.1 Carga y exploración inicial de datos: Se presenta la composición de los datos, identificando la clase principal.
    ◦ 1.2 Limpieza y Preparación de Datos: Detalla la eliminación de columnas no relevantes, ya que no aportan información predictiva o pueden introducir multicolinealidad.
    ◦ 1.3 Análisis de Gráficos para variables numéricas relevantes: Se identifican y analizan variables relevantes, describiendo sus asimetrías y características de distribución. También se incluye el análisis de variables binarias y servicios adicionales, destacando su relación con la probabilidad de abandono y la importancia de algunos servicios.
    ◦ 1.4 Gráficos de variables categóricas. Se identifican y analizan variables categóricas, destacando su relación con cancelación del servicio.
    ◦ 1.5 Información Mutua con Variable Abandono: Se aplica la técnica para medir la dependencia entre dos variables e identificar características por su relevancia, esto nos permitirá simplificar el modelo se ser necesario.
    ◦ 1.6 Detección y tratamiento de outliers: Se aplican transformaciones (logarítmica, Box-Cox y raíz cuadrada) que nos ayudan a Iientificar y transformar valores atípicos que pueden distorsionar el análisis estadístico y el rendimiento de modelos.
    ◦ 1.7 Codificación de variables categóricas: Se identifican las correlaciones positivas y negativas de diversas variables con la probabilidad de abandono.
• 2. Preparación para Modelado
    ◦ 2.1 Separación de Datos: Dividir el conjunto de datos en subconjuntos para entrenamiento, validación y prueba en una proporción 80% entrenamiento, 20% prueba. 
    ◦ 2.2 Balanceo de clases: Se explica la decisión de no aplicar balanceo debido a que la relación de desbalanceo (2.77) se considera moderada y no crítica.
• 3. Entrenamiento de Modelos
    ◦ 3.1 Selección de los Modelos: Se justifica la elección de modelos como Random Forest, CatBoost y XGBoost, destacando sus características y adecuación para el problema.
    ◦ 3.2 Random Forest.
    ◦ 3.3 CatBoost.
    ◦ 3.4 XGBoost.
• 4. Evaluación de los modelos
    ◦ 4.1 Métricas de Random Forest.
    ◦ 4.2 Métricas de CatBoost.
    ◦ 4.3 Métricas de XGBoost.
• 5 Ajuste de modelos
    ◦ 5.1 Random Forest.
    ◦ 5.2 CatBoost.
    ◦ 5.3 XGBoost.
    ◦ 5.4 Evaluación de modelos ajustados.
    ◦ 5.5 Entrenamiento y evaluación en datos de entrenamiento: Se analiza el overfitting de cada modelo, mostrando las diferencias entre el rendimiento en entrenamiento y prueba.
    ◦ 5.6 Validación cruzada.
    ◦ 5.7 Dataframe de variables reducidas: Se describe la estrategia de reducción de variables para disminuir la complejidad, ruido y overfitting. Se presentan las mejoras de rendimiento de Random Forest, CatBoost y XGBoost con variables reducidas, destacando la mejora significativa de XGBoost.
    ◦ 5.8 Ajuste de Umbral: Se realiza un análisis de la curva ROC para optimizar el rendimiento, mostrando que XGBoost logra la mejor precisión y recall.
• 6. Interpretación
    ◦ 6.1 Análisis de las variables relevantes de XGBoost Reducido: Se revisan las variables más relevantes para la predicción de cancelación en el modelo XGBoost reducido y Random Forest reducido, buscando identificar los factores críticos que influyen en la deserción.
    ◦ 6.2 Análisis complementario de variables relevantes Random Forest.
• 7. Anexos.

## Requisitos

- Python 3.x
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

## Uso

1. Clona este repositorio.
2. Abre el archivo `TelecomX_LATAM.ipynb` en Jupyter Notebook o Google Colab.
3. Ejecuta las celdas para reproducir el modelado

---

**Autor:** Ana Sayago  