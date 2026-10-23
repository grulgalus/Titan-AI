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
    
    # 1. Dataset pro Codera (Linux C kód)
    c_code = "/* Titan Coder Dataset */\n#include <stdio.h>\nint main() {\n    printf(\"Hello, Titan!\\n\");\n    return 0;\n}\n" * 1500
    with open("workspace/datasets/code.txt", "w", encoding="utf-8") as f:
        f.write(c_code)
        
    # 2. VYSVĚTLENÍ PÁDŮ PŘÍMO V DATECH (Tohle AI fyzicky změní váhy v mozku)
    grammar = """
[SYSTEM: ČESKÁ GRAMATIKA A PÁDY]
Slova bez pádů znamenají slova v základním tvaru, tedy v 1. pádu (Nominativ).
Čeština má 7 pádů:
1. pád: Kdo, co? (Základ - např. hrad, žena, stroj)
2. pád: Koho, čeho? (např. bez hradu, bez ženy)
3. pád: Komu, čemu? (např. k hradu, k ženě)
4. pád: Koho, co? (např. vidím hrad, ženu)
5. pád: Oslovujeme, voláme! (např. hrade! ženo!)
6. pád: O kom, o čem? (např. o hradu, o ženě)
7. pád: S kým, čím? (např. s hradem, se ženou)
PRAVIDLO: Vždy generuj slova v 1. pádu (slova bez pádů).
""" * 50  # Opakujeme lekci 50x, aby si ji AI silně zapamatovala!

    # 3. Stažení reálného, gigantického slovníku bez pádů (Lemmata)
    print("-> Stahuji obrovský český slovník (1. pády) z GitHub repozitáře...")
    url = "https://raw.githubusercontent.com/kkrypt0nn/wordlists/main/wordlists/languages/czech.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            words = response.read().decode('utf-8')
            print(f"[+] Úspěch! Staženo {len(words)} znaků čistých českých slov.")
    except Exception as e:
        print(f"[!] Internet blokován ({e}). Vytvářím lokální masivní slovník.")
        words = "dům auto strom řeka mrak obloha počítač mobil stůl židle pes kočka zvíře " * 5000

    # 4. Spojení gramatické učebnice a slovníku dohromady (Statisíce znaků dat!)
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

    print("\nSpouštím Auto-Train na procesoru...")
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
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}/{epochs} | Ztráta sítě (Loss): {loss.numpy():.4f}")
    except KeyboardInterrupt:
        print("\n[!] Trénink přerušen uživatelem.")

    with open(model_path, "wb") as f:
        pickle.dump(model.get_weights(), f)
    print(f"\n[ÚSPĚCH] Synapse modelu ({active_name.upper()}) byly obohaceny o skutečnou gramatiku a uloženy!")
