import time
from core.models import MODELS
from memory.state_manager import load_state, save_state
from utils.fs_ops import edit_file_permission

def chat_menu():
    state = load_state()
    print("\n--- [ CHAT MODE ] ---")
    for key, model in MODELS.items():
        print(f"[{key}] {model['name'].upper()} - {model['desc']}")
    
    choice = input("\nVyber model (1-5): ")
    if choice not in MODELS: return
    
    active_model = MODELS[choice]["name"]
    print(f"\nSpuštěn model: {active_model.upper()} (Napiš 'exit' pro konec, 'plan' pro uložení cílů)")

    while True:
        prompt = input(f"\n[{active_model}] Ty: ")
        if prompt.lower() == 'exit': break
        elif prompt.lower() == 'plan':
            plan = input("Zadej nový plán pro vývoj OS/AI: ")
            state["plans"].append(plan)
            state["progress"] += 5
            save_state(state)
            print("Plán úspěšně uložen do paměti.")
            continue

        print(f"[Titan_SI ({active_model}) zpracovává...]")
        time.sleep(0.5)
        
        # Logika Coder modelu
        if active_model == "coder":
            if any(word in prompt.lower() for word in ["make", "vytvoř", "napiš", "os", "kód", "win"]):
                print(f"[{active_model}]: Generuji systémový kód. Vyžaduji potvrzení zápisu.")
                edit_file_permission("kernel.c", "/* Jádro nového OS */\nvoid main() {\n    init_system();\n}")
            else:
                print(f"[{active_model}]: Jsem Coder. Jsem tu na psaní OS, ne na klábosení.")
        else:
            print(f"[{active_model}]: Rozumím. Zde je odpověď na tvůj požadavek: '{prompt}'")
            state["memory"].append({"role": "user", "content": prompt})
            save_state(state)
