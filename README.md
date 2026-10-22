# 🚀 Titan_SI (Super Intelligence Manager)

Titan_SI je pokročilý, modulární AI/SI systém navržený primárně pro běh v prostředí **Termux** na mobilních zařízeních (optimalizováno pro NPU Samsung Galaxy Z Flip 6). Projekt slouží jako interaktivní CLI nástroj pro správu, trénink a komunikaci s různými úrovněmi umělé inteligence.

## ✨ Hlavní funkce

*   **Modulární architektura:** Kód je čistě rozdělen do logických složek (core, memory, utils).
*   **Claude-like Paměť & Plány:** Systém si pamatuje konverzace, ukládá uživatelské plány a sleduje celkový postup (`progress`) vývoje projektu.
*   **Autonomní Coder:** Model `coder` dokáže sám psát kód (např. jádro OS) a ukládat ho do složky `workspace`. Pro bezpečnost vždy vyžaduje administrátorské potvrzení (y/n).
*   **Simulace tréninku na NPU:** Možnost lokálně "trénovat" modely a zvyšovat inteligenci systému.

## 🧠 Dostupné Modely

1.  **Super Flash:** Nejrychlejší model, určený čistě na bleskový pokec.
2.  **Flash:** Zvládá Python a plynulou konverzaci.
3.  **Medium:** Vhodný pro weby, základní assembly a konverzaci.
4.  **Pro:** Nejchytřejší komunikační model, umí základy OS a GUI.
5.  **Coder:** Speciální model s minimem řečí. Zaměřuje se výhradně na programování OS (podobného Windows) a přímou úpravu souborů.

## 📂 Struktura projektu

```text
Titan_SI/
├── main.py                  # Hlavní spouštěcí bod programu
├── README.md                # Dokumentace projektu
├── LICENSE                  # MIT Licence
├── core/                    # Jádro logiky (Modely, Chat, Trénink)
├── memory/                  # Správa paměti a plánů (state.json)
├── workspace/               # Pracovní prostor pro generovaný kód (např. OS)
└── utils/                   # Pomocné skripty (správa oprávnění souborů)
🛠️ Instalace a Spuštění
Otevři aplikaci Termux na svém Androidu.
Ujisti se, že jsi ve složce projektu:
cd ~/Titan_SI
(Volitelné) Spusť instalaci závislostí přímo z menu aplikace, nebo manuálně:
pkg install python clang make -y
Spusť hlavní aplikaci:
python main.py
🎮 Jak používat
Po spuštění main.py se objeví hlavní menu:

[1] Train: Vyber model, který chceš trénovat. Tím se zvýší celkový progress projektu.
[2] Install: Stáhne potřebné Linuxové balíčky do Termuxu pro budoucí kompilaci C/C++ kódů.
[3] Chat: Otevře komunikaci s AI. Během chatu můžeš napsat plan pro uložení tvých cílů do paměti. Pokud vybereš model coder a napíšeš mu ať "vytvoří OS", vyžádá si povolení a vytvoří soubory ve workspace/.
[4] Exit: Bezpečně uloží data a ukončí program.
📜 Licence
Tento projekt je licencován pod MIT License - viz soubor LICENSE pro více detailů. Můžeš ho volně používat, upravovat a sdílet.
