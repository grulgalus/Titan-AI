import os
import platform

def setup_hardware():
    arch = platform.machine().lower()
    is_termux = "PREFIX" in os.environ
    
    print("\n[SYSTÉM] Analyzuji dostupný hardware...")
    
    # 1. EXKLUZIVNÍ SNAPDRAGON (Z Flip 6 / Termux)
    if is_termux and arch == "aarch64":
        print("[!] Detekován ARM64 (Android/Termux).")
        print("[!] Aktivuji BEZPEČNÝ mód pro Snapdragon (Anti-SIGILL)...")
        # Zcela zahodíme C kompilátor, který shazuje systém.
        # Použijeme čistý PYTHON backend. Je neprůstřelný a nepadá!
        os.environ["DEV"] = "PYTHON"
        print("[!] Čistý výpočetní mód odemčen.\n")
        
    # 2. OSTATNÍ ARCHITEKTURY (Tablety / PC)
    else:
        print(f"[!] Detekováno ({arch}), používám standardní CPU backend.")
        os.environ["DEV"] = "CPU"

# Spustíme detektor PŘED načtením TinyGradu!
setup_hardware()

from core.train import train_menu
from core.chat import chat_menu
from core.auto_train import auto_train_menu

def install_menu():
    print("\nInstaluji nástroje pro Termux (TinyGrad, Numpy)...")
    os.system("pip install tinygrad numpy")
    print("\n[ÚSPĚCH] Vše připraveno.")

def main():
    while True:
        print("===============================")
        print(" TITAN SI - TINYGRAD ML ENGINE ")
        print("===============================")
        print("1. Train (Manuální data)")
        print("2. AUTO-TRAIN (Masivní slovníky & kódy)")
        print("3. Chat (Inference)")
        print("4. Install (Nástroje pro Termux)")
        print("5. Exit")
        
        choice = input("Zvol operaci: ")
        if choice == "1": train_menu()
        elif choice == "2": auto_train_menu()
        elif choice == "3": chat_menu()
        elif choice == "4": install_menu()
        elif choice == "5": break
        else: print("Chyba výběru.")

if __name__ == "__main__":
    main()
