import onnx
import onnxruntime
import torch
import numpy as np

# Percorso del tuo modello ONNX
onnx_path = "model.onnx"

# Carica il modello ONNX e controlla che sia valido
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("Modello ONNX valido!")

# Crea sessione ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_path)

# Esempio di input fittizio (uguale a quello usato in export)
dummy_input = np.ones((2, 3, 384, 384), dtype=np.float32)

# Inference
outputs = ort_session.run(None, {"input": dummy_input})

# outputs è una lista di array NumPy (uno per ogni output del modello)
print("Output ONNX shape:", [o.shape for o in outputs])
print(outputs[0])
