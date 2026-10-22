import os

# MODERNÍ TINYGRAD SYNTAXE
# Zcela zahazujeme staré CPU=1, používáme výhradně DEV=CPU
os.environ["DEV"] = "CPU"

from core.train import train_menu
from core.chat import chat_menu
from core.auto_train import auto_train_menu

def install_menu():
    print("\nInstaluji nejnovější TinyGrad a Numpy pro Termux...")
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
        print("4. Install (Stažení TinyGradu)")
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
