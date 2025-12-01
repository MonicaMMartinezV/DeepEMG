import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

baseDir = os.path.dirname(os.path.abspath(__file__))

modelsDir   = os.path.join(baseDir, "Models", "Train models")
scalersDir  = os.path.join(baseDir, "Scalers")
artifactsDir = baseDir

modelPath   = os.path.join(modelsDir, "EMGCNNLSTMModel2.pt")
scalerPath  = os.path.join(scalersDir, "scaler2.save")
curvesPath  = os.path.join(artifactsDir, "training_curves2.npz")

class EMGCNNLSTMClassifier(nn.Module):
    def __init__(self, in_channels=8, num_classes=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

if not os.path.exists(modelPath):
    messagebox.showerror("Error", f"No existe el modelo en: {modelPath}")
    raise SystemExit

if not os.path.exists(scalerPath):
    messagebox.showerror("Error", f"No existe el scaler en: {scalerPath}")
    raise SystemExit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    scaler = joblib.load(scalerPath)
except Exception as e:
    messagebox.showerror("Error", f"No se pudo cargar el scaler: {e}")
    raise

model = EMGCNNLSTMClassifier()
try:
    stateDict = torch.load(modelPath, map_location=device)
    model.load_state_dict(stateDict)
    model.to(device)
    model.eval()
except Exception as e:
    messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
    raise

print("Modelo y scaler cargados correctamente.\nDispositivo:", device)

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

currentFile = None

def predictFromFile():
    global current_file
    if current_file is None:
        messagebox.showwarning("Advertencia", "Primero selecciona un archivo .txt")
        return
    try:
        EMGtxt = pd.read_csv(current_file, sep=r"\s+", engine="python",
                             header=None, on_bad_lines='skip')

        if EMGtxt.shape[1] != 10:
            messagebox.showerror("Error", f"El archivo tiene {EMGtxt.shape[1]} columnas, se esperaban 10.")
            return

        EMGtxt.columns = ['time'] + [f'ch{i}' for i in range(1, 9)] + ['label']
        EMGtxt = EMGtxt.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        print("—" * 60)
        print(f"Archivo: {current_file}")
        print("Conteo de clases:")
        print(EMGtxt['label'].value_counts().sort_index())

        EMGtxt_no_rest = EMGtxt[EMGtxt['label'] != 0]

        if EMGtxt_no_rest.empty:
            messagebox.showinfo("Sin gesto", "Este archivo solo contiene reposo.")
            return

        classPick = int(EMGtxt_no_rest['label'].value_counts().idxmax())

        gesture = EMGtxt_no_rest[EMGtxt_no_rest['label'] == classPick]

        if len(gesture) < 200:
            messagebox.showerror("Error", f"No hay suficientes muestras de la clase {classPick}.")
            return

        channels = [f'ch{i}' for i in range(1, 9)]
        window = gesture.iloc[:200][channels].values

        window_df = pd.DataFrame(window, columns=channels)
        windowScaled = scaler.transform(window_df)

        inputTensor = torch.tensor(windowScaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inputTensor)
            predClass = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        realLabelName = gestureLabels.get(classPick, f"{classPick}")
        predLabelName = gestureLabels.get(predClass, f"{predClass}")

        print("\nProbabilidades:")
        for i, p in enumerate(probs):
            print(f"  clase {i}: {p:.4f}")

        print(f">> Real: {classPick} — {realLabelName}")
        print(f">> Predicha: {predClass} — {predLabelName}")

        result_label.config(
            text=f"Clase Real: {classPick} — {realLabelName}\n"
                 f"Clase Predicha: {predClass} — {predLabelName}"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

def showLearningCurves():
    try:
        data = np.load(curvesPath, allow_pickle=True)
        lossesTrain = data["lossesTrain"]
        accuraciesVal = data["accuraciesVal"]
        accuracyTest = data["accuracyTest"][0]
        maeTest = data["maeTest"][0]
        biasTest = data["biasTest"][0]
        varTest = data["varTest"][0]

        messagebox.showinfo("Métricas del Modelo",
                    f"Accuracy (Test): {accuracyTest:.2f}%\n"
                    f"MAE (Test): {maeTest:.4f}\n"
                    f"Bias (Test): {biasTest:.4f}\n"
                    f"Varianza (Test): {varTest:.4f}")

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(lossesTrain, label="Train Loss", color="blue")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Curva de Loss (Entrenamiento)")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(accuraciesVal, label="Validation Accuracy", color="green")
        plt.axhline(y=accuracyTest, color="red", linestyle="--", label=f"Test Accuracy = {accuracyTest:.2f}%")
        plt.xlabel("Época")
        plt.ylabel("Accuracy (%)")
        plt.title("Curva de Accuracy (Validación y Test)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"No se pudieron cargar curvas: {e}")

root = tk.Tk()
root.title("Clasificación de Gestos EMG - CNN + LSTM")
root.geometry("450x300")

def selectFile():
    global current_file
    file = filedialog.askopenfilename(filetypes=[("Archivos TXT", "*.txt")])
    if file:
        current_file = file
        file_label.config(text=f"Archivo seleccionado:\n{file}")

file_label = tk.Label(root, text="No has seleccionado ningún archivo", wraplength=400)
file_label.pack(pady=10)

btn_select = tk.Button(root, text="Seleccionar archivo .txt", command=selectFile)
btn_select.pack(pady=5)

btn_predict = tk.Button(root, text="Predecir Gesto", command=predictFromFile, bg="green", fg="white")
btn_predict.pack(pady=10)

btn_learning = tk.Button(root, text="Mostrar métricas y curvas de aprendizaje", command=showLearningCurves, bg="purple", fg="white")
btn_learning.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

root.mainloop()