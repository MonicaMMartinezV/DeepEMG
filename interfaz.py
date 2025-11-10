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

class EMGLSTMClassifier(nn.Module):
    def __init__(self, inputSize=8, latentDim=64, layers=1, classes=8, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=inputSize,
                            hidden_size=latentDim,
                            num_layers=layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc1 = nn.Linear(latentDim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)
    

modelPATH = "Modelos/EMGLSTMModel.pt"
scalerPATH = "Scalers/scaler.save"

if not os.path.exists(modelPATH):
    messagebox.showerror("Error", f"No existe el modelo en: {modelPATH}")
    raise SystemExit

if not os.path.exists(scalerPATH):
    messagebox.showerror("Error", f"No existe el scaler en: {scalerPATH}")
    raise SystemExit

scaler = joblib.load(scalerPATH)

try:
    scaler = joblib.load(scalerPATH)
except Exception as e:
    messagebox.showerror("Error", f"No se pudo cargar el scaler: {e}")
    raise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMGLSTMClassifier()
model.load_state_dict(torch.load(modelPATH, map_location=device))
model.to(device)
model.eval()

try:
    model.load_state_dict(torch.load(modelPATH, map_location=device))
except Exception as e:
    messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
    raise

print("Modelo y scaler cargados")

gestureLabels = {
    1: "Mano relajada",
    2: "Puño cerrado",
    3: "Flexión",
    4: "Extensión",
    5: "Desviación radial",
    6: "Desviación cubital",
    7: "Palma extendida"
}

current_file = None

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
        data = np.load("training_curves.npz", allow_pickle=True)
        lossesTrain = data["lossesTrain"]
        accuraciesVal = data["accuraciesVal"]
        accuracyTest = data["accuracyTest"][0]

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
root.title("Clasificación de Gestos EMG - LSTM")
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

btn_learning = tk.Button(root, text="Mostrar curvas de aprendizaje", command=showLearningCurves, bg="purple", fg="white")
btn_learning.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

root.mainloop()