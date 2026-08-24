import json
import re

# Common Gujarati pronouns, stop words, and non-entity nouns
PRONOUNS_AND_NOISE = {
    "હું", "મેં", "તેમ", "મારી", "પોતા", "તે", "માતા", "પિતા", "શાહબુદ્દીન રાઠોડ",
    "મિત્રો", "મને", "મિત્ર", "મહેમાન", "તમે", "તેમણે", "પરિવાર", "જીવન", "પ્રેમ",
    "માનવી", "પુસ્તક", "શાહબુદીન રાઠોડ", "પ્રેમસગાઈ", "મારે", "મારો", "રાઠોડ",
    "માણસ", "શાહબુદીનભાઈ", "મન", "પ્રભુ", "અમે", "આપણને", "તેમની", "આપણા",
    "પોતાના", "તેમણે", "તેના", "તેને", "આપણું", "મારું", "તમારા", "એમણે", "તેમણે",
    "લોકો", "સમાજ", "દુનિયા", "જગત", "ઈશ્વર", "ભગવાન", "વાત", "કામ", "સમય",
    "દિવસ", "વર્ષ", "ઘર", "ગામ", "શાહબુદ્દીન", "શાહબુદીન", "રાઠોડસાહેબ"
}

GENERIC_ROLES = {
    "પુત્ર", "પત્ની", "મહારાજા", "શેઠ", "મંત્રી", "શિક્ષક", "સાહેબ", "રાજા", "દુકાનદાર",
    "ડોક્ટર", "વકીલ", "પોલીસ", "ડ્રાઈવર", "સાધુ", "સંત", "ચોર"
}

with open("oks/oks_characters.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chars = data.get("characters", {})

named_characters = {}
generic_roles = {}
filtered_out = {}

for name, info in chars.items():
    clean_name = name.strip()
    if clean_name in PRONOUNS_AND_NOISE or len(clean_name) <= 1:
        filtered_out[clean_name] = info
    elif clean_name in GENERIC_ROLES:
        generic_roles[clean_name] = info
    else:
        named_characters[clean_name] = info

print(f"Total raw entries: {len(chars)}")
print(f"Filtered out (pronouns/noise): {len(filtered_out)}")
print(f"Generic role titles: {len(generic_roles)}")
print(f"Distinct Named Characters: {len(named_characters)}")

print("\nTop 30 Actual Named Characters:")
sorted_named = sorted(named_characters.items(), key=lambda x: x[1].get("total_appearances", 0), reverse=True)
for k, v in sorted_named[:30]:
    print(f"  - {k}: {v.get('total_appearances', 0)} appearances across {v.get('num_books', 0)} books")
