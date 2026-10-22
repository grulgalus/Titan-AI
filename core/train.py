import time
from core.models import MODELS
from memory.state_manager import load_state, save_state

def train_menu():
    print("\n--- [ TRAIN MODE ] ---")
    for key, model in MODELS.items():
        print(f"[{key}] {model['name'].upper()}")
    
    choice = input("\nVyber model na trénink (1-5): ")
    if choice not in MODELS:
        print("Neplatná volba.")
        return

    m_name = MODELS[choice]["name"]
    print(f"\nIniciuji trénink '{m_name.upper()}' přes NPU Galaxy Z Flip 6...")
    time.sleep(2)
    
    state = load_state()
    state["progress"] += 10
    save_state(state)
    print(f"[ÚSPĚCH] Trénink dokončen. Celkový progress AI: {state['progress']}%")
