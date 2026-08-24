import json
from pathlib import Path

path = Path("oks/oks_characters.json")
if path.exists():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chars = data.get("characters", {})
    sorted_chars = sorted(chars.items(), key=lambda x: x[1].get("total_appearances", 0), reverse=True)
    
    print(f"Total unique raw entries in character index: {len(chars)}")
    print("\nTop 40 extracted 'character' entries:")
    for k, v in sorted_chars[:40]:
        print(f"  - '{k}': {v.get('total_appearances', 0)} appearances across {v.get('num_books', 0)} books")
