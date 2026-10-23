import os
import platform

def setup_hardware():
    arch = platform.machine().lower()
    is_termux = "PREFIX" in os.environ
    
    print("\n[SYSTÉM] Analyzuji dostupný hardware...")
    
    # 1. EXKLUZIVNÍ SNAPDRAGON & ARM64 (TERMUX)
    if is_termux and arch == "aarch64":
        print("[!] Detekován ARM64 (Android/Termux).")
        print("[!] Aktivuji exkluzivní Snapdragon patch (Anti-SIGILL)...")
        # Zabráníme 'illegal hardware instruction' vynucením čisté ARMv8 sady!
        # CFLAGS přepíše výchozí chování Clangu, aby nepoužíval zablokované SVE instrukce.
        os.environ["CFLAGS"] = "-march=armv8-a -O2 -fPIC"
        os.environ["DEV"] = "CPU"
        os.environ["LIBC_PATH"] = "/system/lib64/libc.so"
        print("[!] Výkon mobilního procesoru odemčen.\n")
        
    # 2. KLASICKÝ PC / SERVER (x86_64)
    elif arch in ["x86_64", "amd64"]:
        print("[!] Detekován x86_64 (PC/Server).")
        os.environ["DEV"] = "CPU"
        print("[!] Nastaven klasický CPU backend.\n")
        
    # 3. OSTATNÍ ARCHITEKTURY (Fallback)
    else:
        print(f"[!] Jiná architektura ({arch}), fallback na standardní CPU.")
        os.environ["DEV"] = "CPU"

# Spustíme detektor PŘED načtením TinyGradu!
setup_hardware()

from core.train import train_menu
from core.chat import chat_menu
from core.auto_train import auto_train_menu

def install_menu():
    print("\nInstaluji nástroje pro Termux (TinyGrad, Numpy, Clang)...")
    os.system("pkg install clang -y > /dev/null 2>&1")
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
