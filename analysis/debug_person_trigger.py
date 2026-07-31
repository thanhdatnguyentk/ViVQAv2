import json, re

with open('output/question_entities.json', encoding='utf-8') as f:
    data = json.load(f)

total = len(data)
person_count = sum(1 for v in data.values() if 'person' in v['entities'])
print(f"Total questions   : {total}")
print(f"Has 'person'      : {person_count} ({person_count/total*100:.1f}%)")
print()

# ---- Kiểm tra từng pronoun trong PERSON_PRONOUNS xem cái nào trigger nhiều nhất ----
PERSON_PRONOUNS = [
    "tôi", "tao", "mình", "ta",
    "bạn", "cậu", "mày",
    "anh", "chị", "em", "ông", "bà", "cô", "chú", "bác", "dì", "dượng",
    "anh ấy", "anh ta", "hắn", "y", "gã", "ông ta", "ông ấy",
    "cô ấy", "cô ta", "bà ấy", "chị ấy",
    "nó", "cậu bé", "cô bé", "đứa trẻ", "đứa bé", "em bé",
    "chúng tôi", "chúng ta",
    "mọi người", "các anh", "các chị", "các em", "các ông",
    "hai người", "ba người", "những người", "một người",
    "người đàn ông", "người phụ nữ", "người đó", "người này",
    "ai đó",
]

print("=== Pronoun trigger frequency (substring match) ===")
pronoun_hits = {}
for pronoun in PERSON_PRONOUNS:
    count = sum(1 for v in data.values() if pronoun in v['question'].lower())
    pronoun_hits[pronoun] = count

for p, c in sorted(pronoun_hits.items(), key=lambda x: -x[1])[:20]:
    print(f"  '{p}' → triggers in {c} questions")

print()
print("=== False positive examples ===")
# Kiem tra 'ong' trong 'khong'
print(f"'ông' in 'không' (substring) : {'ông' in 'không'}")
print(f"'bà' in 'bàn'  (substring)   : {'bà' in 'bàn'}")
print(f"'cô' in 'công'  (substring)  : {'cô' in 'công'}")
print(f"'nó' in 'nóng'  (substring)  : {'nó' in 'nóng'}")
print(f"'chú' in 'chúng' (substring) : {'chú' in 'chúng'}")
print(f"'ta' in 'tay'   (substring)  : {'ta' in 'tay'}")
