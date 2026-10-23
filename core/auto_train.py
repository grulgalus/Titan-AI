import os
import urllib.request
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
    print("\n-> Připravuji masivní tréninková data...")
    
    grammar = """
[SYSTEM: ČESKÁ GRAMATIKA A PÁDY]
Slova bez pádů znamenají slova v základním tvaru, tedy v 1. pádu (Nominativ).
Čeština má 7 pádů:
1. pád: Kdo, co? (Základ - např. hrad, žena, stroj)
2. pád: Koho, čeho?
3. pád: Komu, čemu?
4. pád: Koho, co?
5. pád: Oslovujeme, voláme!
6. pád: O kom, o čem?
7. pád: S kým, čím?
PRAVIDLO: Vždy generuj slova v 1. pádu (slova bez pádů).
""" * 50

    print("-> Stahuji obrovský český slovník (1. pády) z GitHub repozitáře...")
    url = "https://raw.githubusercontent.com/kkrypt0nn/wordlists/main/wordlists/languages/czech.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            words = response.read().decode('utf-8')
            print(f"[+] Úspěch! Staženo {len(words)} znaků čistých slov.")
    except Exception as e:
        print(f"[!] Internet blokován ({e}).")
        words = "dům auto strom řeka mrak obloha počítač mobil stůl židle pes " * 5000

    full_text = (grammar + "\n[SLOVNÍK]:\n" + words[:150000]) * 3
    with open("workspace/datasets/czech.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

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
    print("\n--- [ AUTO PRE-TRAIN ] ---")
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

    model = TitanLLM()
    model_path = f"memory/{active_name}.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model.load_weights(pickle.load(f))

    optimizer = Adam([model.w1, model.b1, model.w2, model.b2], lr=0.01)

    print("\nSpouštím Auto-Train (Výpočet matic)...")
    epochs = 400
    try:
        with Context(TRAINING=1):
            for epoch in range(1, epochs + 1):
                X_batch, Y_batch = get_batch(text, BATCH_SIZE)
                out = model.forward(X_batch)
                loss = (out.log_softmax() * Y_batch).sum().div(BATCH_SIZE).mul(-1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # OPRAVA: Vypisujeme Ztrátu (Loss) KAŽDOU JEDNU EPOCHU!
                # Tím donutíme grafickou/výpočetní kartu okamžitě vydat výsledek a nezamrzne to.
                print(f"Epoch {epoch}/{epochs} | Ztráta sítě (Loss): {loss.numpy():.4f}")
                
    except KeyboardInterrupt:
        print("\n[!] Trénink přerušen uživatelem.")

    with open(model_path, "wb") as f:
        pickle.dump(model.get_weights(), f)
    print(f"\n[ÚSPĚCH] Synapse byly uloženy!")
