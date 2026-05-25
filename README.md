# **DeepEMG — Clasificación de gestos de mano mediante señales EMG con modelos LSTM, CNN-LSTM y BiLSTM**

Este proyecto implementa un sistema completo para la clasificación automática de gestos de mano a partir de señales electromiográficas (EMG) adquiridas con el brazalete Myo Armband.

El sistema emplea distintas arquitecturas de redes neuronales profundas desarrolladas en PyTorch:

* **Modelo 1:** LSTM
* **Modelo 2:** CNN + LSTM
* **Modelo 3:** BiLSTM + Data Augmentation + Rebalanceo de clases
* **Modelo 4:** BiLSTM con pipeline corregido, normalización sin fuga de datos, split por archivo y rebalanceo controlado

El objetivo del proyecto es comparar el rendimiento de diferentes modelos de deep learning para clasificación de gestos EMG, evaluar métricas clave y proporcionar una interfaz funcional para predicciones en tiempo real. Además, el proyecto incluye una versión corregida del pipeline de datos para evitar fuga de información, aplicar normalización únicamente con datos de entrenamiento y evaluar el modelo con una partición más representativa de todas las clases.

## Descripción general

El sistema procesa archivos de texto (.txt) que contienen lecturas de 8 canales EMG obtenidas mediante un brazalete Myo Armband [Dataset](https://doi.org/10.24432/C5ZP5C). 

Cada archivo representa una sesión de captura de señales musculares correspondientes a diferentes gestos de la mano, tales como flexión, extensión, puño cerrado o desviaciones radial y cubital.

## Principales funcionalidades

* Lectura, limpieza y normalización de señales EMG multicanal.
* Segmentación con sliding windows (200×8).
* Generación de ventanas por archivo para respetar la estructura original de cada sesión.
* Normalización sin fuga de datos, ajustando el `StandardScaler` únicamente con el conjunto de entrenamiento.
* División train/validation/test con cobertura de todas las clases.
* Data augmentation aplicado únicamente sobre el conjunto de entrenamiento.
* Rebalanceo mediante aumento dirigido y pesos automáticos por clase.
* Entrenamiento reproducible mediante PyTorch.
* Evaluación integral (Accuracy, MAE, Bias, Varianza, Matriz de Confusión, Precision, Recall y F1-score).
* Exportación automática del conjunto de prueba a una carpeta `TestCases/`.
* Interfaces gráficas (Tkinter) para cargar archivos `.txt` y obtener predicciones.
* Reporte con resultados de cada modelo y comparativa final entre las versiones desarrolladas.

## Resumen de modelos

| Métrica | Modelo 1 (LSTM) | Modelo 2 (CNN+LSTM) | Modelo 3 (BiLSTM+Augmentation+rebalanceo) | Modelo 4 (BiLSTM + pipeline corregido) |
|---|---:|---:|---:|---:|
| **Accuracy (Test)** | 69.94% | 74.38% | 79.62% | **86.36%** |
| **MAE (Test)** | 1.0109 | 0.8282 | 0.6412 | **0.2664** |
| **Bias (Test)** | -0.0570 | -0.1222 | 0.0114 | **-0.0603** |
| **Varianza (Test)** | 3.9233 | 3.7161 | 3.6433 | **3.0080** |

El Modelo 4 se posiciona como la versión más robusta del proyecto. A diferencia de los modelos anteriores, esta versión corrige el pipeline de preparación de datos para reducir fuga de información, genera ventanas por archivo, ajusta la normalización únicamente con el conjunto de entrenamiento, aplica data augmentation solo sobre train y utiliza una partición que garantiza presencia de todas las clases en train, validation y test.

## Carpeta TestCases

En los modelos se añadió un módulo de código para:

- Identificar automáticamente qué ventanas pertenecen al conjunto de prueba
- Copiar los archivos `.txt` originales a la carpeta `TestCases/`

Esto permite probar fácilmente los modelos desde las interfaces sin mezclar datos de entrenamiento.

## Ejecutar el proyecto

### *Requisito importante:* Descargar el dataset y configurar el entorno

Este proyecto no incluye el dataset original por razones de tamaño y licencia.
Para que los notebooks y las interfaces funcionen correctamente, es necesario:

#### 1. Descargar el dataset manualmente

Dataset oficial:
[https://doi.org/10.24432/C5ZP5C](https://doi.org/10.24432/C5ZP5C)

Descarga los archivos `.txt` y colócalos en una carpeta llamada:

```
DeepEMG/Dataset/
```

El repositorio no funcionará hasta que el dataset esté presente.

#### 2. Ajustar rutas del proyecto según tu entorno

Los notebooks usan rutas como:

```
/content/drive/MyDrive/ConcentracionIA/Periodo2/Mod2. Tecnicas y arquitecturas de deep learning/Proyecto/DeepEMG
```

Si ejecutas el proyecto:

* En Google Colab debes montar tu Drive y modificar `baseDir` al path correcto donde guardaste el proyecto
* En tu computadora cambia `baseDir` a la ubicación donde se clonó el repositorio

Por ejemplo:

```python
baseDir = "/content/drive/MyDrive/TuCarpeta/DeepEMG"
# o en ejecución local:
baseDir = "C:/Users/TuUsuario/Documents/DeepEMG"
```

Si esta ruta no coincide con la ubicación real del dataset, modelos o escaladores, el código marcará error al intentar cargar archivos.

### Google Colab

1. Abrir cualquiera de los notebooks:
   * `Model1.ipynb`
   * `Model2.ipynb`
   * `Model3.ipynb`
   * `Model4.ipynb`
2. Montar Google Drive
3. Ejecutar todas las celdas
4. Los pesos, curvas y artefactos se guardarán automáticamente
5. Usar la celda de inferencia para cargar archivos `.txt`

## Ejecución de interfaz gráfica

### 1. Clonar repositorio

```bash
git clone https://github.com/MonicaMMartinezV/DeepEMG.git
cd DeepEMG
```

### 2. Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Si PyTorch no se instala correctamente desde `requirements.txt`, puede instalarse manualmente con:

```bash
pip install torch torchvision torchaudio
```

### 3. Ejecutar una de las interfaces

```bash
python Interface1.py   # LSTM
python Interface2.py   # CNN+LSTM
python Interface3.py   # BiLSTM
python Interface4.py   # BiLSTM con pipeline corregido
```

Las GUI permiten:

* Seleccionar un archivo `.txt`
* Ver la clase real detectada
* Obtener la predicción del modelo
* Visualizar métricas y curvas de entrenamiento

## Dataset

Dataset oficial utilizado:
**Krilova et al., EMG Data for Gestures (2018)**
UCI Machine Learning Repository
[https://doi.org/10.24432/C5ZP5C](https://doi.org/10.24432/C5ZP5C)

Contiene:
- 8 canales EMG
- Sesiones completas por sujeto
- Etiquetas originales de 0 a 7
- Formato `.txt` estandarizado

En este proyecto, la etiqueta `0` se elimina durante el preprocesamiento porque representa segmentos sin gesto activo o reposo. Por ello, los modelos entrenan con las etiquetas activas `1` a `7`, que internamente se codifican como clases `0` a `6` para ser compatibles con `CrossEntropyLoss`.

## Tecnologías

* Python 3.10 / 3.11 / 3.12
* * PyTorch
* NumPy / Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Tkinter
* Google Colab / Drive

## Autora

**Mónica Monserrat Martínez Vásquez**

*Tec de Monterrey*

*[A01710965@tec.mx](mailto:A01710965@tec.mx)*

Deep Learning · IA Biomédica · Procesamiento de Señales EMG

## Backup
Este es un backup de proyecto entero en drive, en caso de perdida.
[backup drive](https://drive.google.com/drive/folders/1H5ZihJImpYQyOi0dm4xr23-Oq8IHfhaY?usp=sharing)

## Referencias

[1] Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S. (2018). *EMG Data for Gestures* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5ZP5C](https://doi.org/10.24432/C5ZP5C).
