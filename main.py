import os
from core.train import train_menu
from core.chat import chat_menu

def install_menu():
    print("\n--- [ INSTALL MODE ] ---")
    print("Připravuji prostředí pro Titan_SI...")
    os.system("pkg install python clang make -y > /dev/null 2>&1")
    print("Instalace balíčků hotova.")

def main():
    while True:
        print("\n===============================")
        print("      TITAN SI - NEXUS v2.0    ")
        print("===============================")
        print("1. Train (Trénink modelů)")
        print("2. Install (Stažení závislostí)")
        print("3. Chat (Spuštění AI)")
        print("4. Exit")
        
        choice = input("Zvol operaci: ")
        if choice == "1": train_menu()
        elif choice == "2": install_menu()
        elif choice == "3": chat_menu()
        elif choice == "4":
            print("Ukládám neurální sítě... Vypínám Titan_SI.")
            break
        else:
            print("Chyba: Neplatný vstup.")

if __name__ == "__main__":
    main()
