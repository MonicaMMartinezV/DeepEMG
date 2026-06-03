import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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

modelPath  = os.path.join(modelsDir, "EMGCNNBiLSTMAttentionClassifierModel4.pt")
scalerPath = os.path.join(scalersDir, "scaler4.save")
curvesPath = os.path.join(artifactsDir, "training_curves4.npz")

class EMGCNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, in_channels=8, num_classes=7):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        logits = self.fc(context)
        return logits

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

    model = EMGCNNBiLSTMAttentionClassifier(
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

    model = EMGCNNBiLSTMAttentionClassifier(
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

        confidence = np.max(probs) * 100

        result_label.config(
            text=f"{predOriginalLabel} — {predLabelName}"
        )

        real_label_value.config(
            text=f"{realOriginalLabel} — {realLabelName}"
        )

        confidence_label.config(
            text=f"{confidence:.2f}%"
        )

        confidence_detail_label.config(
            text=f"El modelo predijo esta clase con una confianza del {confidence:.2f}%."
        )

        confidence_bar["value"] = confidence

        status_label.config(
            text="Predicción realizada correctamente.",
            foreground=SUCCESS_COLOR
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
        lossFinalTrain = data["lossFinalTrain"][0] if "lossFinalTrain" in data else None

        metric_text = ""

        if accuracyTest is not None:
            metric_text += f"Accuracy (Test): {accuracyTest:.2f}%\n"
        if maeTest is not None:
            metric_text += f"MAE (Test): {maeTest:.4f}\n"
        if biasTest is not None:
            metric_text += f"Bias (Test): {biasTest:.4f}\n"
        if varTest is not None:
            metric_text += f"Varianza (Test): {varTest:.4f}\n"
        if lossFinalTrain is not None:
            metric_text += f"Loss final (Train): {lossFinalTrain:.4f}\n"

        if not metric_text:
            metric_text = "No se encontraron métricas de test en el archivo."

        messagebox.showinfo("Métricas del Modelo 4", metric_text)

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

BG_COLOR = "#F7F2FF"
CARD_COLOR = "#FFFFFF"
PRIMARY_COLOR = "#7C3AED"
PRIMARY_DARK = "#5B21B6"
PRIMARY_LIGHT = "#EDE9FE"
SUCCESS_COLOR = "#6D28D9"
TEXT_COLOR = "#1F1235"
MUTED_COLOR = "#6B5A7A"
BORDER_COLOR = "#DDD6FE"

root = tk.Tk()
root.title("DeepEMG — CNN + BiLSTM + Attention")
root.geometry("750x700")
root.resizable(False, False)
root.configure(bg=BG_COLOR)

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Primary.TButton",
    font=("Segoe UI", 10, "bold"),
    padding=10,
    background=PRIMARY_COLOR,
    foreground="white",
    borderwidth=0
)

style.map(
    "Primary.TButton",
    background=[("active", PRIMARY_DARK)]
)

style.configure(
    "Outline.TButton",
    font=("Segoe UI", 10, "bold"),
    padding=10,
    background=PRIMARY_LIGHT,
    foreground=PRIMARY_DARK,
    borderwidth=0
)

style.map(
    "Outline.TButton",
    background=[("active", BORDER_COLOR)]
)

style.configure(
    "Purple.Horizontal.TProgressbar",
    thickness=15,
    troughcolor="#EDE9FE",
    background=PRIMARY_COLOR,
    bordercolor="#EDE9FE",
    lightcolor=PRIMARY_COLOR,
    darkcolor=PRIMARY_COLOR
)


def create_card(parent, padx=26, pady=12):
    card = tk.Frame(
        parent,
        bg=CARD_COLOR,
        highlightbackground=BORDER_COLOR,
        highlightthickness=1
    )
    card.pack(fill="x", padx=padx, pady=pady)
    return card


def selectFile():
    global current_file

    file = filedialog.askopenfilename(
        title="Selecciona un archivo EMG",
        filetypes=[("Archivos TXT", "*.txt")]
    )

    if file:
        current_file = file
        file_name = os.path.basename(file)

        file_label.config(
            text=file_name,
            fg=TEXT_COLOR
        )

        file_path_label.config(
            text=file,
            fg=MUTED_COLOR
        )

        result_label.config(text="Sin predicción")
        real_label_value.config(text="—")
        confidence_label.config(text="—")
        confidence_detail_label.config(
            text="Selecciona un archivo y presiona “Predecir gesto” para ver el resultado."
        )
        confidence_bar["value"] = 0

        status_label.config(
            text="Archivo cargado correctamente. Listo para predecir.",
            fg=SUCCESS_COLOR
        )

header = tk.Frame(root, bg=PRIMARY_COLOR, height=100)
header.pack(fill="x")
header.pack_propagate(False)

header_left = tk.Frame(header, bg=PRIMARY_COLOR)
header_left.pack(side="left", fill="both", expand=True, padx=28, pady=16)

title_label = tk.Label(
    header_left,
    text="DeepEMG",
    font=("Segoe UI", 25, "bold"),
    bg=PRIMARY_COLOR,
    fg="white"
)
title_label.pack(anchor="w")

subtitle_label = tk.Label(
    header_left,
    text="Clasificación de gestos EMG con CNN + BiLSTM + Attention",
    font=("Segoe UI", 10),
    bg=PRIMARY_COLOR,
    fg="#EDE9FE"
)
subtitle_label.pack(anchor="w", pady=(2, 0))

header_right = tk.Frame(header, bg=PRIMARY_COLOR)
header_right.pack(side="right", padx=28, pady=24)

btn_learning = ttk.Button(
    header_right,
    text="Ver métricas",
    command=showLearningCurves,
    style="Outline.TButton"
)
btn_learning.pack()

main_container = tk.Frame(root, bg=BG_COLOR)
main_container.pack(fill="both", expand=True)

file_card = create_card(main_container)

file_title = tk.Label(
    file_card,
    text="1. Seleccionar archivo EMG",
    font=("Segoe UI", 13, "bold"),
    bg=CARD_COLOR,
    fg=TEXT_COLOR
)
file_title.pack(anchor="w", padx=18, pady=(14, 2))

file_description = tk.Label(
    file_card,
    text="Carga un archivo .txt con señales EMG para realizar la predicción del gesto.",
    font=("Segoe UI", 9),
    bg=CARD_COLOR,
    fg=MUTED_COLOR
)
file_description.pack(anchor="w", padx=18, pady=(0, 10))

file_label = tk.Label(
    file_card,
    text="No has seleccionado ningún archivo",
    font=("Segoe UI", 10, "bold"),
    bg=CARD_COLOR,
    fg=MUTED_COLOR,
    wraplength=670,
    justify="left"
)
file_label.pack(anchor="w", padx=18, pady=(0, 2))

file_path_label = tk.Label(
    file_card,
    text="",
    font=("Segoe UI", 8),
    bg=CARD_COLOR,
    fg=MUTED_COLOR,
    wraplength=680,
    justify="left"
)
file_path_label.pack(anchor="w", padx=18, pady=(0, 12))

button_row = tk.Frame(file_card, bg=CARD_COLOR)
button_row.pack(anchor="w", padx=18, pady=(0, 16))

btn_select = ttk.Button(
    button_row,
    text="Seleccionar archivo",
    command=selectFile,
    style="Primary.TButton"
)
btn_select.pack(side="left", padx=(0, 10))

btn_predict = ttk.Button(
    button_row,
    text="Predecir gesto",
    command=predictFromFile,
    style="Primary.TButton"
)
btn_predict.pack(side="left")


result_card = create_card(main_container)

result_title = tk.Label(
    result_card,
    text="2. Resultado de la predicción",
    font=("Segoe UI", 13, "bold"),
    bg=CARD_COLOR,
    fg=TEXT_COLOR
)
result_title.pack(anchor="w", padx=18, pady=(14, 10))

prediction_grid = tk.Frame(result_card, bg=CARD_COLOR)
prediction_grid.pack(fill="x", padx=18, pady=(0, 8))

predicted_label_title = tk.Label(
    prediction_grid,
    text="Gesto predicho",
    font=("Segoe UI", 9),
    bg=CARD_COLOR,
    fg=MUTED_COLOR
)
predicted_label_title.grid(row=0, column=0, sticky="w")

result_label = tk.Label(
    prediction_grid,
    text="Sin predicción",
    font=("Segoe UI", 20, "bold"),
    bg=CARD_COLOR,
    fg=PRIMARY_COLOR
)
result_label.grid(row=1, column=0, sticky="w", pady=(2, 12))

real_label_title = tk.Label(
    prediction_grid,
    text="Clase real detectada en el archivo",
    font=("Segoe UI", 9),
    bg=CARD_COLOR,
    fg=MUTED_COLOR
)
real_label_title.grid(row=2, column=0, sticky="w")

real_label_value = tk.Label(
    prediction_grid,
    text="—",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_COLOR,
    fg=TEXT_COLOR
)
real_label_value.grid(row=3, column=0, sticky="w", pady=(2, 12))

confidence_title = tk.Label(
    prediction_grid,
    text="Precisión / confianza de la predicción",
    font=("Segoe UI", 9),
    bg=CARD_COLOR,
    fg=MUTED_COLOR
)
confidence_title.grid(row=4, column=0, sticky="w")

confidence_row = tk.Frame(prediction_grid, bg=CARD_COLOR)
confidence_row.grid(row=5, column=0, sticky="ew", pady=(4, 4))

confidence_bar = ttk.Progressbar(
    confidence_row,
    orient="horizontal",
    length=460,
    mode="determinate",
    maximum=100,
    style="Purple.Horizontal.TProgressbar"
)
confidence_bar.pack(side="left", padx=(0, 12))

confidence_label = tk.Label(
    confidence_row,
    text="—",
    font=("Segoe UI", 12, "bold"),
    bg=CARD_COLOR,
    fg=PRIMARY_DARK
)
confidence_label.pack(side="left")

confidence_detail_label = tk.Label(
    prediction_grid,
    text="Selecciona un archivo y presiona “Predecir gesto” para ver el resultado.",
    font=("Segoe UI", 9),
    bg=CARD_COLOR,
    fg=MUTED_COLOR,
    wraplength=670,
    justify="left"
)
confidence_detail_label.grid(row=6, column=0, sticky="w", pady=(2, 12))

status_frame = tk.Frame(root, bg=BG_COLOR)
status_frame.pack(fill="x", padx=26, pady=(0, 14))

status_label = tk.Label(
    status_frame,
    text=f"Modelo cargado correctamente en {device}.",
    font=("Segoe UI", 9),
    bg=BG_COLOR,
    fg=MUTED_COLOR
)
status_label.pack(anchor="w")

root.mainloop()