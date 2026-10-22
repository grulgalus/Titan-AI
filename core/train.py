import os
import pickle
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from core.model import TitanLLM

AVAILABLE_MODELS = ["super_flash", "flash", "medium", "pro", "coder"]
CONTEXT_SIZE = 8

def train_menu():
    print("\n--- [ TINYGRAD TRAIN MODE ] ---")
    for i, name in enumerate(AVAILABLE_MODELS, 1):
        print(f"[{i}] {name.upper()}")
    
    choice = input("\nVyber model k přetrénování vah (1-5): ")
    try:
        active_name = AVAILABLE_MODELS[int(choice) - 1]
    except:
        return print("Neplatná volba.")

    print(f"\n[Trénink modelu {active_name.upper()}]")
    text = input("Napiš data pro AI (např. kus C kódu): ")
    if len(text) < CONTEXT_SIZE + 2:
        text = text.ljust(CONTEXT_SIZE + 2, " ")

    # Inicializace modelu
    model = TitanLLM()
    model_path = f"memory/{active_name}.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model.load_weights(pickle.load(f))
        print("Načteny předchozí synapse z paměti...")

    optimizer = Adam([model.w1, model.b1, model.w2, model.b2], lr=0.01)

    print("\nSpouštím Backpropagation přes TinyGrad na procesoru...")
    
    # Příprava datasetu (One-Hot Encoding znaků)
    X, Y = [], []
    for i in range(len(text) - CONTEXT_SIZE):
        vec = np.zeros(CONTEXT_SIZE * 256, dtype=np.float32)
        for j, char in enumerate(text[i:i+CONTEXT_SIZE]):
            vec[j * 256 + (ord(char) % 256)] = 1.0
        X.append(vec)
        Y.append(ord(text[i+CONTEXT_SIZE]) % 256)
        
    X_tensor = Tensor(X)
    batch_size = len(Y)
    
    # Maska pro výpočet chyby
    Y_mask = np.zeros((batch_size, 256), dtype=np.float32)
    Y_mask[np.arange(batch_size), Y] = 1.0
    Y_mask_t = Tensor(Y_mask)

    # 100 Epcoh učení
    for epoch in range(1, 101):
        out = model.forward(X_tensor)
        
        # Ztrátová funkce (Cross Entropy)
        loss = (out.log_softmax() * Y_mask_t).sum().div(batch_size).mul(-1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/100 | Ztráta (Loss): {loss.numpy():.4f}")

    # Uložení vah
    with open(model_path, "wb") as f:
        pickle.dump(model.get_weights(), f)
    print(f"\n[ÚSPĚCH] Nové váhy úspěšně uloženy do {model_path}!")
