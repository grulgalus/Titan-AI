import os

def setup_hardware():
    # Zjistíme výrobce zařízení přímo ze systému Android
    manufacturer = os.popen("getprop ro.product.manufacturer").read().strip().lower()
    
    print("\n[SYSTÉM] Analyzuji hardware...")
    
    if "samsung" in manufacturer:
        print(f"[!] Detekován {manufacturer.capitalize()} (Snapdragon).")
        print("[!] Ochrana aktivní: Používám bezpečný PYTHON backend.")
        os.environ["DEV"] = "PYTHON"
    else:
        print(f"[!] Detekováno Lenovo/Tablet -> ARMv8.")
        print("[!] Výkon odemčen: Používám nativní CPU backend.")
        os.environ["DEV"] = "CPU"
        os.environ["LIBC_PATH"] = "/system/lib64/libc.so"

setup_hardware()

from core.train import train_menu
from core.chat import chat_menu
from core.auto_train import auto_train_menu

def install_menu():
    print("\nInstaluji nástroje pro Termux...")
    os.system("pip install tinygrad numpy")
    print("\n[ÚSPĚCH] Hotovo.")

def main():
    while True:
        print("\n===============================")
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
        else: print("Chyba.")

if __name__ == "__main__":
    main()
