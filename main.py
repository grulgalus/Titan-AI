import os

# MODERNÍ TINYGRAD SYNTAXE (Vynucení CPU)
os.environ["DEV"] = "CPU"

# OPRAVA PRO TERMUX/ANDROID:
# Říkáme TinyGradu, ať nehledá klasický Linuxový glibc, ale použije Androidí Bionic libc!
os.environ["LIBC_PATH"] = "/system/lib64/libc.so"

from core.train import train_menu
from core.chat import chat_menu
from core.auto_train import auto_train_menu

def install_menu():
    print("\nInstaluji nástroje pro Termux (TinyGrad, Numpy, Clang)...")
    # TinyGrad potřebuje kompilátor (Clang), aby mohl za běhu tvořit matice
    os.system("pkg install clang -y > /dev/null 2>&1")
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
