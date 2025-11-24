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

modelPath   = os.path.join(modelsDir, "EMGBiLSTMModel3.pt")
scalerPath  = os.path.join(scalersDir, "scaler.save")
curvesPath  = os.path.join(artifactsDir, "training_curves3.npz")

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

model = EMGBiLSTMClassifier()
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
    1: "Mano relajada",
    2: "Puño cerrado",
    3: "Flexión",
    4: "Extensión",
    5: "Desviación radial",
    6: "Desviación cubital",
    7: "Palma extendida"
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
        EMGtxt = EMGtxt[EMGtxt['label'] != 0].reset_index(drop=True)
        EMGtxt = EMGtxt.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        print("—"*60)
        print(f"Archivo: {current_file}")
        print("Clases presentes (únicas) y conteos:")
        print(EMGtxt['label'].value_counts().sort_index())

        classCounts = EMGtxt['label'].value_counts()
        classCounts = classCounts[classCounts.index != 0]

        classPick = classCounts.idxmax()
        gesture = EMGtxt[EMGtxt['label'] == classPick]

        if len(gesture) < 200:
            messagebox.showinfo("Sin suficientes datos", f"Se necesitan 200 muestras, solo hay {len(gesture)}.")
            return

        channels = [f'ch{i}' for i in range(1, 9)]
        window = gesture.iloc[:200][channels]
        windowScaled = scaler.transform(window)  
        inputTensor = torch.tensor(windowScaled, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inputTensor)
            predClass = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        
        predClass = predClass + 1
        predClass = int(predClass)
        classPick = int(classPick)
        
        predLabel = gestureLabels.get(predClass, f"{predClass}")
        realLabel = gestureLabels.get(classPick, f"{classPick}")

        

        print("Probabilidades (softmax) por clase índice (0..7):")
        for i, p in enumerate(probs):
            print(f"  clase {i}: {p:.4f}")
        print(f">> Predicción índice: {predClass}  →  {predLabel}")
        print(f">> Clase real (mayoritaria en archivo): {classPick}  →  {realLabel}")
        print("—"*60)

        result_label.config(text=f"Clase Real: {classPick} — {realLabel}\n"
                                 f"Clase Predicha: {predClass} — {predLabel}")

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

        # Loss
        plt.subplot(1,2,1)
        plt.plot(lossesTrain, label="Train Loss", color="blue")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Curva de Loss (Entrenamiento)")
        plt.grid(True)

        # Validation Accuracy
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