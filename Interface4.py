import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

baseDir = os.path.dirname(os.path.abspath(__file__))

modelsDir = os.path.join(baseDir, "Models", "Train models")
scalersDir = os.path.join(baseDir, "Scalers")
artifactsDir = baseDir

modelPath = os.path.join(modelsDir, "EMGBiLSTMModel4.pt")
scalerPath = os.path.join(scalersDir, "scaler4.save")
curvesPath = os.path.join(artifactsDir, "training_curves4.npz")

class EMGBiLSTMClassifier(nn.Module):
    def __init__(self, in_channels=8, num_classes=8):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_checkpoint_safely(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibilidad con versiones anteriores de PyTorch
        return torch.load(path, map_location=device)


def normalize_index_to_label(raw_index_to_label):
    if raw_index_to_label is None:
        return None

    fixed = {}
    for k, v in raw_index_to_label.items():
        fixed[int(k)] = int(v)
    return fixed


def normalize_label_to_index(raw_label_to_index):
    if raw_label_to_index is None:
        return None

    fixed = {}
    for k, v in raw_label_to_index.items():
        fixed[int(k)] = int(v)
    return fixed

if not os.path.exists(modelPath):
    raise FileNotFoundError(f"No existe el modelo en: {modelPath}")

if not os.path.exists(scalerPath):
    raise FileNotFoundError(f"No existe el scaler en: {scalerPath}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = joblib.load(scalerPath)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el scaler: {e}")


checkpoint = load_checkpoint_safely(modelPath, device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    numClasses = int(checkpoint.get("numClasses", 8))
    windowSize = int(checkpoint.get("windowSize", 200))
    stepSize = int(checkpoint.get("stepSize", 100))

    labelToIndex = normalize_label_to_index(checkpoint.get("labelToIndex", None))
    indexToLabel = normalize_index_to_label(checkpoint.get("indexToLabel", None))

    model = EMGBiLSTMClassifier(
        in_channels=8,
        num_classes=numClasses
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Checkpoint completo cargado correctamente.")

else:
    numClasses = 8
    windowSize = 200
    stepSize = 100
    labelToIndex = None
    indexToLabel = None

    model = EMGBiLSTMClassifier(
        in_channels=8,
        num_classes=numClasses
    ).to(device)

    model.load_state_dict(checkpoint)

    print("State_dict directo cargado correctamente.")

model.to(device)
model.eval()

print("Modelo y scaler cargados correctamente.")
print("Dispositivo:", device)
print("Número de clases:", numClasses)
print("Window size:", windowSize)
print("Step size:", stepSize)
print("labelToIndex:", labelToIndex)
print("indexToLabel:", indexToLabel)

gestureLabels = {
    0: "Mano en reposo",
    1: "Puño cerrado",
    2: "Flexión",
    3: "Extensión",
    4: "Desviación radial",
    5: "Desviación cubital",
    6: "Palma extendida",
    7: "Otro / transición / gesto variable"
}

def get_gesture_name(original_label):
    return gestureLabels.get(int(original_label), f"Clase {original_label}")

current_file = None


def preprocess_file_for_prediction(file_path):
    EMGtxt = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python",
        header=None,
        on_bad_lines="skip"
    )

    if EMGtxt.shape[1] != 10:
        raise ValueError(
            f"El archivo tiene {EMGtxt.shape[1]} columnas, pero se esperaban 10."
        )

    EMGtxt.columns = ["time"] + [f"ch{i}" for i in range(1, 9)] + ["label"]

    EMGtxt = EMGtxt.apply(pd.to_numeric, errors="coerce")
    EMGtxt = EMGtxt.dropna().reset_index(drop=True)

    if EMGtxt.empty:
        raise ValueError("El archivo quedó vacío después de limpiar valores no numéricos.")

    return EMGtxt


def choose_prediction_window(EMGtxt):
    channels = [f"ch{i}" for i in range(1, 9)]

    non_rest = EMGtxt[EMGtxt["label"] != 0].copy()

    if non_rest.empty:
        if len(EMGtxt) < windowSize:
            raise ValueError(
                f"El archivo solo contiene reposo y tiene menos de {windowSize} muestras."
            )

        real_label = int(EMGtxt["label"].value_counts().idxmax())
        window = EMGtxt.iloc[:windowSize][channels].values
        return window, real_label

    real_label = int(non_rest["label"].value_counts().idxmax())
    gesture_df = non_rest[non_rest["label"] == real_label].copy()

    if len(gesture_df) < windowSize:
        raise ValueError(
            f"No hay suficientes muestras de la clase {real_label}. "
            f"Se necesitan {windowSize}, pero hay {len(gesture_df)}."
        )

    window = gesture_df.iloc[:windowSize][channels].values

    return window, real_label


def predictFromFile():
    global current_file

    if current_file is None:
        messagebox.showwarning("Advertencia", "Primero selecciona un archivo .txt")
        return

    try:
        EMGtxt = preprocess_file_for_prediction(current_file)

        print("—" * 60)
        print(f"Archivo: {current_file}")
        print("Conteo de clases en archivo:")
        print(EMGtxt["label"].value_counts().sort_index())

        window, realOriginalLabel = choose_prediction_window(EMGtxt)

        channels = [f"ch{i}" for i in range(1, 9)]
        windowDf = pd.DataFrame(window, columns=channels)

        windowScaled = scaler.transform(windowDf)

        inputTensor = torch.tensor(
            windowScaled,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inputTensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predIndex = int(torch.argmax(logits, dim=1).item())

        if indexToLabel is not None:
            predOriginalLabel = int(indexToLabel[predIndex])
        else:
            predOriginalLabel = predIndex

        realLabelName = get_gesture_name(realOriginalLabel)
        predLabelName = get_gesture_name(predOriginalLabel)

        print("\nProbabilidades por índice interno del modelo:")
        for i, p in enumerate(probs):
            if indexToLabel is not None:
                original_label = indexToLabel.get(i, i)
            else:
                original_label = i

            label_name = get_gesture_name(original_label)
            print(f"  índice {i} / label original {original_label} ({label_name}): {p:.4f}")

        print(f">> Real: {realOriginalLabel} — {realLabelName}")
        print(f">> Predicha: {predOriginalLabel} — {predLabelName}")

        result_label.config(
            text=(
                f"Clase real: {realOriginalLabel} — {realLabelName}\n"
                f"Clase predicha: {predOriginalLabel} — {predLabelName}\n"
                f"Confianza: {np.max(probs) * 100:.2f}%"
            )
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print("ERROR:", e)

def showLearningCurves():
    if not os.path.exists(curvesPath):
        messagebox.showerror(
            "Error",
            f"No existe el archivo de curvas en:\n{curvesPath}"
        )
        return

    try:
        data = np.load(curvesPath, allow_pickle=True)

        lossesTrain = data["lossesTrain"]
        accuraciesVal = data["accuraciesVal"]

        accuracyTest = data["accuracyTest"][0] if "accuracyTest" in data else None
        maeTest = data["maeTest"][0] if "maeTest" in data else None
        biasTest = data["biasTest"][0] if "biasTest" in data else None
        varTest = data["varTest"][0] if "varTest" in data else None

        metric_text = ""

        if accuracyTest is not None:
            metric_text += f"Accuracy (Test): {accuracyTest:.2f}%\n"
        if maeTest is not None:
            metric_text += f"MAE (Test): {maeTest:.4f}\n"
        if biasTest is not None:
            metric_text += f"Bias (Test): {biasTest:.4f}\n"
        if varTest is not None:
            metric_text += f"Varianza (Test): {varTest:.4f}\n"

        if not metric_text:
            metric_text = "No se encontraron métricas de test en el archivo."

        messagebox.showinfo("Métricas del Modelo", metric_text)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(lossesTrain, label="Train Loss")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Curva de Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuraciesVal, label="Validation Accuracy")

        if accuracyTest is not None:
            plt.axhline(
                y=accuracyTest,
                linestyle="--",
                label=f"Test Accuracy = {accuracyTest:.2f}%"
            )

        plt.xlabel("Época")
        plt.ylabel("Accuracy (%)")
        plt.title("Curva de Accuracy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"No se pudieron cargar curvas: {e}")
        print("ERROR curvas:", e)

root = tk.Tk()
root.title("Clasificación de Gestos EMG - BiLSTM")
root.geometry("520x340")


def selectFile():
    global current_file

    file = filedialog.askopenfilename(
        filetypes=[("Archivos TXT", "*.txt")]
    )

    if file:
        current_file = file
        file_label.config(text=f"Archivo seleccionado:\n{file}")
        result_label.config(text="")


file_label = tk.Label(
    root,
    text="No has seleccionado ningún archivo",
    wraplength=480
)
file_label.pack(pady=10)

btn_select = tk.Button(
    root,
    text="Seleccionar archivo .txt",
    command=selectFile
)
btn_select.pack(pady=5)

btn_predict = tk.Button(
    root,
    text="Predecir gesto",
    command=predictFromFile,
    bg="green",
    fg="white"
)
btn_predict.pack(pady=10)

btn_learning = tk.Button(
    root,
    text="Mostrar métricas y curvas de aprendizaje",
    command=showLearningCurves,
    bg="purple",
    fg="white"
)
btn_learning.pack(pady=10)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 12),
    fg="blue",
    wraplength=480
)
result_label.pack(pady=10)

root.mainloop()