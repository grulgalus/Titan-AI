import os
import pickle
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.helpers import Context
from core.model import TitanLLM

AVAILABLE_MODELS = ["super_flash", "flash", "medium", "pro", "coder"]
CONTEXT_SIZE = 8
BATCH_SIZE = 64

def fetch_data():
    os.makedirs("workspace/datasets", exist_ok=True)
    print("-> Připravuji masivní tréninková data...")
    c_code = "/* Titan Coder Dataset */\n#include <stdio.h>\nint main() {\n    printf(\"Hello, Titan OS!\\n\");\n    return 0;\n}\n" * 1500
    with open("workspace/datasets/code.txt", "w", encoding="utf-8") as f:
        f.write(c_code)
    cz_text = "Ahoj, já jsem Titan_SI. Jsem tvá umělá inteligence. Dnes je krásný den. Funguji bezchybně offline na telefonu. Rád programuji a analyzuji data. Jsem Super Flash a odpovídám velmi rychle. " * 1500
    with open("workspace/datasets/czech.txt", "w", encoding="utf-8") as f:
        f.write(cz_text)

def get_batch(text, batch_size):
    ix = np.random.randint(0, len(text) - CONTEXT_SIZE - 1, batch_size)
    X = np.zeros((batch_size, CONTEXT_SIZE * 256), dtype=np.float32)
    Y = np.zeros((batch_size, 256), dtype=np.float32)
    for i, idx in enumerate(ix):
        for j, char in enumerate(text[idx:idx+CONTEXT_SIZE]):
            X[i, j * 256 + (ord(char) % 256)] = 1.0
        Y[i, ord(text[idx+CONTEXT_SIZE]) % 256] = 1.0
    return Tensor(X), Tensor(Y)

def auto_train_menu():
    print("\n--- [ AUTO PRE-TRAIN (MASIVNÍ UČENÍ) ] ---")
    for i, name in enumerate(AVAILABLE_MODELS, 1):
        print(f"[{i}] {name.upper()}")
    choice = input("\nVyber model (1-5): ")
    try:
        active_name = AVAILABLE_MODELS[int(choice) - 1]
    except:
        return print("Neplatná volba.")

    fetch_data()
    dataset_path = "workspace/datasets/code.txt" if active_name == "coder" else "workspace/datasets/czech.txt"
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[!] Dataset načten: {len(text)} znaků k analýze.")

    model = TitanLLM()
    model_path = f"memory/{active_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model.load_weights(pickle.load(f))

    optimizer = Adam([model.w1, model.b1, model.w2, model.b2], lr=0.01)

    print("\nSpouštím Auto-Train na CPU (Termux Compatible)...")
    epochs = 400
    try:
        # TADY JE TEN FIX: Povolujeme TinyGradu měnit váhy neuronů
        with Context(TRAINING=1):
            for epoch in range(1, epochs + 1):
                X_batch, Y_batch = get_batch(text, BATCH_SIZE)
                out = model.forward(X_batch)
                loss = (out.log_softmax() * Y_batch).sum().div(BATCH_SIZE).mul(-1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}/{epochs} | Ztráta sítě (Loss): {loss.numpy():.4f}")
    except KeyboardInterrupt:
        print("\n[!] Trénink byl přerušen uživatelem.")

    with open(model_path, "wb") as f:
        pickle.dump(model.get_weights(), f)
    print(f"\n[ÚSPĚCH] Nové synapse ({active_name.upper()}) byly trvale uloženy!")
