import os
import time
import pickle
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from core.model import TitanLLM

AVAILABLE_MODELS = ["super_flash", "flash", "medium", "pro", "coder"]
CONTEXT_SIZE = 8
BATCH_SIZE = 64  # Učíme 64 vzorů najednou, ať to na NPU odsýpá!

def download_datasets():
    print("\n[DATA FETCH] Připojuji se k repozitářům pro stažení tréninkových dat...")
    os.makedirs("workspace/datasets", exist_ok=True)
    
    # Stažení masivního českého textu/slovníku (pokud ještě není)
    if not os.path.exists("workspace/datasets/czech.txt"):
        print("-> Stahuji masivní český slovník a korpus...")
        # Použijeme curl ke stažení raw dat
        os.system("curl -s -L https://raw.githubusercontent.com/david-p/czech-wordlist/master/czech-wordlist.txt -o workspace/datasets/czech.txt")
        # Přidáme nějaké české věty pro strukturu
        os.system("echo '\nAhoj, jak se máš? Já jsem umělá inteligence.\nToto je český jazyk a já se učím mluvit.\n' >> workspace/datasets/czech.txt")

    # Stažení programátorských kódů pro Codera (Linux Kernel C kód a Python)
    if not os.path.exists("workspace/datasets/code.txt"):
        print("-> Stahuji zdrojové kódy (Linux Kernel C & Python)...")
        os.system("curl -s -L https://raw.githubusercontent.com/torvalds/linux/master/init/main.c -o workspace/datasets/code.txt")
        os.system("curl -s -L https://raw.githubusercontent.com/tinygrad/tinygrad/master/tinygrad/tensor.py >> workspace/datasets/code.txt")
        
    print("[DATA FETCH] Všechna tréninková data jsou lokálně uložena!\n")

def get_batch(text, batch_size):
    # Generátor náhodných dávek dat (aby RAM nespadla na obřím souboru)
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
    
    choice = input("\nVyber model k absolutnímu přetrénování (1-5): ")
    try:
        active_name = AVAILABLE_MODELS[int(choice) - 1]
    except:
        return print("Neplatná volba.")

    download_datasets()

    # Vybereme správný dataset podle modelu
    dataset_path = "workspace/datasets/code.txt" if active_name == "coder" else "workspace/datasets/czech.txt"
    
    print(f"Načítám dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    print(f"[!] Dataset načten: {len(text)} znaků k analýze.")

    # Inicializace neuronové sítě
    model = TitanLLM()
    model_path = f"memory/{active_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model.load_weights(pickle.load(f))
        print("Načteny předchozí synapse. Budeme je vylepšovat!")

    optimizer = Adam([model.w1, model.b1, model.w2, model.b2], lr=0.005)

    print("\nSpouštím Auto-Train (Backpropagation po dávkách).")
    print("VAROVÁNÍ: Procesor Z Flip 6 teď pojede naplno! Může se zahřát.")
    print("Pro zastavení zmáčkni CTRL+C.\n")

    epochs = 500 # Kolikrát se váhy přepočítají
    try:
        for epoch in range(1, epochs + 1):
            X_batch, Y_batch = get_batch(text, BATCH_SIZE)
            
            # Forward pass
            out = model.forward(X_batch)
            
            # Loss (Cross Entropy) a Backward pass
            loss = (out.log_softmax() * Y_batch).sum().div(BATCH_SIZE).mul(-1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{epochs} | Ztráta sítě (Loss): {loss.numpy():.4f}")
                
    except KeyboardInterrupt:
        print("\n[!] Trénink manuálně přerušen uživatelem.")

    # Uložení nového, chytrého mozku!
    with open(model_path, "wb") as f:
        pickle.dump(model.get_weights(), f)
    print(f"\n[ÚSPĚCH] Fyzická struktura mozku ({active_name.upper()}) byla upravena a uložena!")
