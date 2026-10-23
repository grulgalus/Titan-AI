import os
import pickle
import numpy as np
from tinygrad.tensor import Tensor
from core.model import TitanLLM

AVAILABLE_MODELS = ["super_flash", "flash", "medium", "pro", "coder"]
CONTEXT_SIZE = 8

def chat_menu():
    print("\n--- [ TINYGRAD INFERENCE MODE ] ---")
    for i, name in enumerate(AVAILABLE_MODELS, 1):
        print(f"[{i}] {name.upper()}")
    
    choice = input("\nVyber model (1-5): ")
    try:
        active_name = AVAILABLE_MODELS[int(choice) - 1]
    except:
        return print("Neplatná volba.")

    model = TitanLLM()
    model_path = f"memory/{active_name}.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model.load_weights(pickle.load(f))
        print(f"[!] Načteny natrénované váhy: {active_name.upper()}")
    else:
        print("[VAROVÁNÍ] Model nemá natrénované váhy! Bude psát odpad.")

    while True:
        prompt = input(f"\n[{active_name}] Ty: ")
        if prompt.lower() == 'exit': break

        # Vyplníme kontext mezerami, pokud je moc krátký
        context = prompt[-CONTEXT_SIZE:].ljust(CONTEXT_SIZE, " ")
        
        print(f"[{active_name} generuje...]: ", end="", flush=True)
        
        for _ in range(100):
            # OPRAVA PRO TABLET: Vytvoříme rovnou čistou 2D matici
            vec = np.zeros((1, CONTEXT_SIZE * 256), dtype=np.float32)
            for i, char in enumerate(context):
                vec[0, i * 256 + (ord(char) % 256)] = 1.0
                
            x = Tensor(vec)
            logits = model.forward(x).numpy()[0]
            
            # Vyhodnocení (0.5 je nižší kreativita = více se drží toho, co se naučil)
            temp = 0.5 
            logits_max = np.max(logits)
            exp_preds = np.exp((logits - logits_max) / temp)
            preds = exp_preds / np.sum(exp_preds)
            
            # Náhodný výběr podle pravděpodobností
            char_idx = np.random.choice(256, p=preds)
            char = chr(char_idx)
            
            print(char, end="", flush=True)
            context = context[1:] + char
        print()
