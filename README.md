# Telecom X - Predicción de Cancelación de Clientes (Churn) - Parte 2

## 📋 Descripción del Proyecto

Este repositorio contiene la segunda parte del desafío **Telecom X**, cuyo objetivo es construir modelos predictivos de machine learning para identificar clientes con alta probabilidad de cancelar sus servicios (churn). A partir de un dataset previamente limpiado y estandarizado, se realiza un pipeline completo que incluye:

- Preparación de datos (encoding, balanceo de clases, normalización).
- Análisis de correlación y visualizaciones dirigidas.
- Entrenamiento y evaluación de múltiples modelos (Regresión Logística y Random Forest, con y sin balanceo SMOTE).
- Interpretación de resultados e identificación de los principales factores que influyen en la cancelación.
- Propuestas de estrategias de retención basadas en los insights obtenidos.

## 🎯 Objetivo Principal

Desarrollar un modelo predictivo robusto que permita a Telecom X anticiparse a la cancelación de clientes, identificando los factores clave que inciden en el churn y proporcionando recomendaciones accionables para el área de retención.

## 📁 Estructura del Proyecto

```
.
├── datos.csv                      # Archivo de datos tratado (Parte 1)
├── Telecom2.ipynb                 # Cuaderno Jupyter con todo el análisis y modelado
└── README.md                      # Este archivo
```

> **Nota:** El archivo `datos.csv` debe contener las mismas columnas y calidad de datos que se generaron en la Parte 1 del desafío (limpieza y estandarización).

## 🛠️ Proceso de Preparación de los Datos

### 1. Carga y eliminación de columnas irrelevantes
- Se carga el archivo `datos.csv` y se elimina la columna `ID_Cliente` por ser un identificador único sin valor predictivo.

### 2. Codificación de variables categóricas
- Se identifican las columnas categóricas y se clasifican en binarias (mapeo directo `Yes`/`No` → 1/0) y multicategóricas (one-hot encoding con `drop_first=True`).
- La columna `Genero` (Female/Male) se mapea a 0/1.
- Las columnas booleanas generadas por `pd.get_dummies` se convierten a enteros (0/1).

### 3. Verificación de desbalance de clases
- La variable objetivo `Evasion` presenta un 73.4% de clientes que no cancelan y 26.6% que sí cancelan. Existe un desbalance moderado.

### 4. Balanceo con SMOTE
- Se aplica **SMOTE** (Synthetic Minority Over-sampling Technique) sobre el conjunto de entrenamiento para equilibrar las clases (50% cada una), evitando contaminar el conjunto de prueba.

### 5. Normalización / Estandarización
- Para modelos sensibles a la escala (Regresión Logística), se utiliza `StandardScaler` dentro de un pipeline.
- Modelos basados en árboles (Random Forest) no requieren normalización.

## 📊 Análisis Exploratorio y de Correlación

- Se calcula la matriz de correlación y se identifican las variables con mayor correlación con `Evasion`:
  - Positivas: `Servicio_Internet_Fiber optic`, `Metodo_Pago_Electronic check`, `Cuentas_Diarias`, `Cargos_Mensuales`.
  - Negativas: `Meses_Contrato`, `Tipo_Contrato_Two year`, `Cargos_Totales` (interesante porque en modelos tiene signo positivo, indicando relaciones no lineales).

- **Análisis dirigido**:
  - Clientes que cancelan tienen **menor tiempo de contrato** (mediana ~10 meses vs ~40 meses).
  - Distribución de `Cargos_Totales` muestra que los que cancelan tienen valores más bajos (contradice correlación lineal, debido a la interacción con antigüedad).

## 🤖 Modelos Predictivos

Se entrenan cuatro modelos:

| Modelo               | Descripción                                  |
|----------------------|----------------------------------------------|
| RL (original)        | Regresión Logística con normalización (datos originales) |
| RF (original)        | Random Forest sin normalización (datos originales) |
| RL + SMOTE           | Regresión Logística con datos balanceados    |
| RF + SMOTE           | Random Forest con datos balanceados           |

### Evaluación y métricas

| Modelo               | Exactitud | Precisión | Recall | F1-score | Overfitting |
|----------------------|-----------|-----------|--------|----------|-------------|
| RL (original)        | 80.2%     | 65.4%     | 54.2%  | 0.593    | No          |
| RF (original)        | 79.2%     | 63.9%     | 50.1%  | 0.561    | Sí (dif >0.2)|
| RL + SMOTE           | 77.1%     | 56.0%     | 65.1%  | 0.602    | No          |
| RF + SMOTE           | 76.7%     | 55.8%     | 59.4%  | 0.575    | Sí (dif >0.2)|

- **Mejor F1-score:** RL + SMOTE (0.602) → equilibrio entre precisión y recall.
- **Mejor recall:** RL + SMOTE (65.1%) → identifica más clientes que realmente cancelan.
- **Problema de overfitting:** Random Forest memoriza los datos de entrenamiento; se recomienda ajustar hiperparámetros (max_depth, min_samples_split).

## 🔍 Importancia de Variables

### Regresión Logística (coeficientes)
- Positivos (aumentan probabilidad de churn): `Servicio_Internet_Fiber optic`, `Cargos_Totales`, `Metodo_Pago_Electronic check`, `Facturacion_Electronica`.
- Negativos (disminuyen churn): `Meses_Contrato`, `Tipo_Contrato_Two year`, `Tipo_Contrato_One year`, `Soporte_Tecnico_Yes`.

### Random Forest (importancia)
- Variables más relevantes: `Cargos_Totales`, `Meses_Contrato`, `Cargos_Mensuales`, `Cuentas_Diarias`, `Metodo_Pago_Electronic check`.

### Hallazgos clave
- Clientes con **contratos más largos** y que usan **soporte técnico** son menos propensos a cancelar.
- **Fibra óptica** y **pago con cheque electrónico** se asocian a mayor churn.
- El **gasto total** tiene una relación compleja: aunque correlación negativa, en modelos ajustados aparece como positivo, sugiriendo que clientes con alto gasto acumulado (y posiblemente mayor antigüedad) tienen riesgo si no están satisfechos.

## 📈 Conclusiones y Estrategias de Retención

1. **Incentivar contratos de largo plazo** (descuentos por anualidad, beneficios exclusivos).
2. **Mejorar la experiencia de clientes con fibra óptica** (encuestas de satisfacción, calidad de servicio).
3. **Segmentar y contactar clientes con alto gasto acumulado** para ofrecer programas de fidelización.
4. **Promover métodos de pago automáticos** (domiciliación, tarjeta) en lugar de cheque electrónico.
5. **Potenciar servicios de valor agregado** (soporte técnico, seguridad online) como factores protectores del churn.

## 🚀 Instrucciones para Ejecutar el Cuaderno

### Requisitos previos
- Python 3.8 o superior.
- Instalar las bibliotecas necesarias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### Pasos
1. Clona este repositorio o descarga los archivos.
2. Asegúrate de que el archivo `datos.csv` (tratado en la Parte 1) esté en el mismo directorio que el cuaderno.
3. Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
jupyter notebook Telecom2.ipynb
```

4. Ejecuta todas las celdas en orden. Los resultados (gráficos, tablas, métricas) se mostrarán directamente en el cuaderno.

## 📚 Dependencias Principales

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn