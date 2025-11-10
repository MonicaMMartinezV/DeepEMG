# DeepEMG
Clasificación de gestos de mano mediante señales EMG utilizando redes LSTM

Este proyecto implementa un sistema de clasificación automática de gestos de mano a partir de señales electromiográficas (EMG) multicanal, empleando redes neuronales LSTM (Long Short-Term Memory).  

El modelo fue desarrollado en PyTorch, y entrenado tanto en Google Colab como en un entorno local de Python, asegurando portabilidad, reproducibilidad y compatibilidad entre versiones.

---

## Descripción general

El sistema procesa archivos de texto (.txt) que contienen lecturas de 8 canales EMG obtenidas mediante un brazalete Myo Armband [Dataset](https://doi.org/10.24432/C5ZP5C). 

Cada archivo representa una sesión de captura de señales musculares correspondientes a diferentes gestos de la mano, tales como flexión, extensión, puño cerrado o desviaciones radial y cubital.

---

## Entornos de ejecución

El proyecto puede ejecutarse de dos formas equivalentes:

### **Versión Colab**
Notebook interactivo con visualización y widgets.
- Archivos: `Modelo.ipynb` `Interfaz.ipynb`
- Monta Google Drive, entrena el modelo, genera métricas y visualiza curvas.
- Incluye un widget para subir archivos .txt y ejecutar predicciones.

### **Versión Local (Python .py)**
Ejecutable de escritorio con interfaz Tkinter.
- Archivos principales:
  - `modelo.py`: Entrenamiento y guardado del modelo.
  - `interfaz.py`: Carga del modelo y ejecución de inferencia con GUI.
- Permite seleccionar un archivo, predecir el gesto dominante y visualizar las curvas de aprendizaje en una ventana separada.

---

## Uso en Google Colab

1. Abre los notebook `Modelo.ipynb` `Interfaz.ipynb` en Google Colab.
2. Monta tu Google Drive (`drive.mount('/content/drive')`).
3. Ejecuta todas las celdas hasta el entrenamiento.
4. Las curvas de aprendizaje se guardan automáticamente como `training_curves.npz`.
5. Sube un archivo .txt desde el widget de carga para obtener una predicción en tiempo real.

## Uso e instalación version .py

1. Clona el repositorio:

```bash
git clone https://github.com/MonicaMMartinezV/DeepEMG.git
cd DeepEMG
```

2. Crea un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate     # (Linux/Mac)
.venv\Scripts\activate        # (Windows)

pip install -r requirements.txt
```

3. Inicia la interfaz:

```bash
python interfaz.py
```

---

**Mónica Monserrat Martínez Vásquez**

*Tec de Monterrey*

*[A01710965@tec.mx](mailto:A01710965@tec.mx)*

---

## Referencias

[1]	Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S. (2018). EMG Data for Gestures (Dataset). UCI Machine Learning Repository. [https://doi.org/10.24432/C5ZP5C](https://doi.org/10.24432/C5ZP5C).
