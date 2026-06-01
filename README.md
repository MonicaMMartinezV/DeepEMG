# **DeepEMG — Clasificación de gestos de mano mediante señales EMG con modelos LSTM, CNN-LSTM, BiLSTM y CNN-BiLSTM-Attention**

Este proyecto implementa un sistema completo para la clasificación automática de gestos de mano a partir de señales electromiográficas (EMG) adquiridas con el brazalete Myo Armband.

El sistema emplea distintas arquitecturas de redes neuronales profundas desarrolladas en PyTorch:

* **Modelo 1:** LSTM
* **Modelo 2:** CNN + LSTM
* **Modelo 3:** BiLSTM + Data Augmentation + Rebalanceo de clases
* **Modelo 4:** CNN + BiLSTM + Attention

El objetivo del proyecto es comparar el rendimiento de diferentes modelos de deep learning para clasificación de gestos EMG, evaluar métricas clave y proporcionar una interfaz funcional para predicciones en tiempo real. Además, el proyecto incluye una versión corregida del pipeline de datos para evitar fuga de información, aplicar normalización únicamente con datos de entrenamiento y evaluar el modelo con una partición más representativa de todas las clases.

## Descripción general

El sistema procesa archivos de texto (.txt) que contienen lecturas de 8 canales EMG obtenidas mediante un brazalete Myo Armband [Dataset](https://doi.org/10.24432/C5ZP5C). 

Cada archivo representa una sesión de captura de señales musculares correspondientes a diferentes gestos de la mano, tales como flexión, extensión, puño cerrado o desviaciones radial y cubital.

## Principales funcionalidades

* Lectura, limpieza y normalización de señales EMG multicanal.
* Eliminación de segmentos con etiqueta `0`, correspondientes a reposo o ausencia de gesto activo.
* Segmentación con sliding windows (200×8).
* Generación de ventanas por archivo para respetar la estructura original de cada sesión.
* Codificación de etiquetas originales `1–7` a clases internas `0–6`, compatibles con `CrossEntropyLoss`.
* Normalización sin fuga de datos, ajustando el `StandardScaler` únicamente con el conjunto de entrenamiento.
* División train/validation/test con cobertura de todas las clases.
* Data augmentation aplicado únicamente sobre el conjunto de entrenamiento.
* Rebalanceo mediante aumento dirigido y pesos automáticos por clase.
* Entrenamiento reproducible mediante PyTorch.
* Comparación de arquitecturas LSTM, CNN + LSTM, BiLSTM y CNN + BiLSTM + Attention.
* Evaluación integral (Accuracy, MAE, Bias, Varianza, Loss final de entrenamiento, Matriz de Confusión, Precision, Recall y F1-score).
* Guardado del mejor checkpoint según validation accuracy.
* Guardado del scaler y curvas de entrenamiento.
* Interfaces gráficas (Tkinter) para cargar archivos `.txt` y obtener predicciones.
* Reporte con resultados de cada modelo y comparativa final entre las versiones desarrolladas.
* Carpeta `TestCases/` para almacenar archivos `.txt` utilizados en pruebas desde la interfaz.

## Resumen de modelos

| Métrica | Modelo 1 (LSTM) | Modelo 2 (CNN + LSTM) | Modelo 3 (BiLSTM + Augmentation + Rebalanceo) | Modelo 4 (CNN + BiLSTM + Attention) |
|---|---:|---:|---:|---:|
| **Accuracy (Test)** | 69.94% | 74.38% | 79.62% | **89.34%** |
| **MAE (Test)** | 1.0109 | 0.8282 | 0.6412 | **0.2125** |
| **Bias (Test)** | -0.0570 | -0.1222 | 0.0114 | **-0.0580** |
| **Varianza (Test)** | 3.9233 | 3.7161 | 3.6433 | **3.1413** |
| **Loss final (Train)** | 0.4037 | 0.3124 | 0.0992 | **0.0148** |

#### Modelo 4: CNN + BiLSTM + Attention

El Modelo 4 integra tres componentes principales:

1. CNN 1D, para extraer patrones locales y relaciones entre los canales EMG.
2. BiLSTM, para capturar dependencias temporales hacia adelante y hacia atrás.
3. Attention, para asignar mayor peso a los instantes temporales más relevantes dentro de cada ventana.

El **Modelo 4 (CNN + BiLSTM + Attention)** se posiciona como la arquitectura con mejor desempeño global del proyecto, alcanzando un **89.34% de accuracy en test**, un **MAE de 0.2125**, un **bias de -0.0580**, una **varianza de 3.1413** y una **loss final de entrenamiento de 0.0148**. Esta mejora se debe a la integración de extracción espacial mediante CNN, modelado temporal bidireccional mediante BiLSTM y un mecanismo de atención que permite ponderar los segmentos temporales más relevantes de cada ventana EMG.

En el reporte de clasificación, el Modelo 4 obtuvo un **weighted F1-score de 0.8931** y un **macro F1-score de 0.8752**. Además, a diferencia de corridas anteriores, la clase 7 sí estuvo presente en el conjunto de prueba con 28 muestras, alcanzando un **F1-score de 0.7463**.

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
python Interface4.py   # CNN + BiLSTM + Attention
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

Durante el preprocesamiento se elimina la etiqueta `0`, correspondiente a reposo o ausencia de gesto activo. Por ello, el modelo trabaja con las etiquetas originales `1–7`, las cuales son codificadas internamente como `0–6` para ser compatibles con `CrossEntropyLoss`.

## Tecnologías

* Python 3.10 / 3.11 / 3.12
* PyTorch
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

[2] CNN-LSTM model for Time Series Forecasting. (n.d.). EmergentMind. [https://www.emergentmind.com/topics/cnn-lstm-model-for-time-series-forecasting](https://www.emergentmind.com/topics/cnn-lstm-model-for-time-series-forecasting).

[3] De Jonge, S., Potters, W. V., & Verhamme, C. (2024). Artificial intelligence for automatic classification of needle EMG signals: A scoping review. Clinical Neurophysiology, 159, 41–55. [https://doi.org/10.1016/j.clinph.2023.12.134](https://doi.org/10.1016/j.clinph.2023.12.134).

[4] Meta. (n.d.). EMG Wristbands and Technology. Meta Emerging Tech. [https://www.meta.com/emerging-tech/emg-wearable-technology/](https://www.meta.com/emerging-tech/emg-wearable-technology/).

[5] Joshi, D. C., Kumar, P., Joshi, R. C., & Mitra, S. (2024). AI-enhanced analysis to investigate the feasibility of EMG signals for prosthetic hand force control incorporating anthropometric measures. Prosthesis, 6(6), 1459–1478. [https://doi.org/10.3390/prosthesis6060106](https://doi.org/10.3390/prosthesis6060106).

[6] L. Wang, J. Fu, B. Zheng and H. Zhao, "Research on sEMG–based gesture recognition using the Attention-based LSTM-CNN with Stationary Wavelet Packet Transform," 2022 4th International Conference on Advances in Computer Technology, Information Science and Communications (CTISC), Suzhou, China, 2022, pp. 1-6, [https://doi.org/10.1109/CTISC54888.2022.9849743](https://doi.org/10.1109/CTISC54888.2022.9849743).

[7] J. Shin, A. S. M. Miah, S. Konnai, S. Hoshitaka and P. Kim, "Electromyography-Based Gesture Recognition With Explainable AI (XAI): Hierarchical Feature Extraction for Enhanced Spatial-Temporal Dynamics," in IEEE Access, vol. 13, pp. 88930-88951, 2025, [https://doi.org/10.1109/ACCESS.2025.3569899](https://doi.org/10.1109/ACCESS.2025.3569899).

[8] W. Ma, G. Song, Q. Zeng, H. Zhang, M. Zou and Z. Zhao, "FFCSLT: A Deep Learning Model for Traffic Police Hand Gesture Recognition Using Surface Electromyographic Signals," in IEEE Sensors Journal, vol. 24, no. 8, pp. 13640-13655, 15 April15, 2024, [https://doi.org/10.1109/JSEN.2024.3371588](https://doi.org/10.1109/JSEN.2024.3371588).

[9] J. Guo and Z. Li, "A Continuous Hand Movement Recognition Method for sEMG Signals Based on BiLSTM Network and Attention Mechanism," 2023 5th International Conference on Robotics, Intelligent Control and Artificial Intelligence (RICAI), Hangzhou, China, 2023, pp. 1160-1164, [https://doi.org/10.1109/RICAI60863.2023.10489184](https://doi.org/10.1109/RICAI60863.2023.10489184).

[10] H. Le, G. M. Spinks, M. i. h. Panhuis and G. Alici, "Cross-Day Myoelectric Gesture Recognition with Hybrid Multistream CNN-Bidirectional LSTM," 2025 IEEE International Conference on Mechatronics (ICM), Wollongong, Australia, 2025, pp. 1-6, [https://doi.org/10.1109/ICM62621.2025.10934890](https://doi.org/10.1109/ICM62621.2025.10934890).



