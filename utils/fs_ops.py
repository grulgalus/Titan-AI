def edit_file_permission(filename, content):
    print(f"\n[UPOZORNĚNÍ] Model 'coder' chce vytvořit/upravit soubor: workspace/{filename}")
    auth = input("Povolit tuto akci (Admin práva)? (y/n): ")
    if auth.lower() == 'y':
        with open(f"workspace/{filename}", "w") as f:
            f.write(content)
        print(f"[ÚSPĚCH] Systémový soubor '{filename}' byl zapsán.")
    else:
        print("[ZAMÍTNUTO] Přístup odepřen uživatelem.")
