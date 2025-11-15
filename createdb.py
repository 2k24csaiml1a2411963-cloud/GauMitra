import sqlite3

conn = sqlite3.connect("cows.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS breeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    origin TEXT,
    color TEXT,
    milk_yield TEXT,
    characteristics TEXT
)
""")


breeds_data = [
    ("Alambadi", "Tamil Nadu, India", "Grey or white", "Low to medium", "Hardy draught breed, adapted to hilly terrain"),
    ("Amritmahal", "Karnataka, India", "Grey", "Medium", "Strong bullocks used for ploughing and carting"),
    ("Ayrshire", "Scotland", "Red and white", "High (6000+ L/year)", "Excellent dairy breed with good udder conformation"),
    ("Banni", "Kutch, Gujarat", "Brown or white", "Medium to high", "Good heat tolerance and high milk fat content"),
    ("Bargur", "Tamil Nadu, India", "Reddish brown with white markings", "Low", "Known for speed and endurance on slopes"),
    ("Bhadawari", "Uttar Pradesh, India", "Copper-colored", "1200–1500 L/year", "High fat milk content (up to 8%)"),
    ("Brown_Swiss", "Switzerland", "Brown or grey", "6000–9000 L/year", "Calm temperament, high milk yield"),
    ("Dangi", "Maharashtra, India", "Grey with spots", "Low", "Resistant to heavy rainfall, used for draught"),
    ("Deoni", "Maharashtra, India", "White with black spots", "1200–1800 L/year", "Dual-purpose (milk and draught) breed"),
    ("Gir", "Gujarat, India", "Red with white patches", "1200–1800 L/year", "High milk yield and disease resistance"),
    ("Guernsey", "Channel Islands", "Fawn with white markings", "5500–7000 L/year", "Milk rich in beta-carotene"),
    ("Hallikar", "Karnataka, India", "Grey", "Low", "Strong draught breed for heavy work"),
    ("Hariana", "Haryana, India", "White or light grey", "1000–1500 L/year", "Dual-purpose, hardy and adaptable"),
    ("Holstein_Friesian", "Netherlands", "Black and white", "8000–10000 L/year", "World’s highest milk-producing breed"),
    ("Jaffrabadi", "Gujarat, India", "Black", "1500–2000 L/year", "Heavy, strong buffalo breed"),
    ("Jersey", "Jersey Island", "Light brown", "5000–8000 L/year", "High butterfat milk content, small body size"),
    ("Kangayam", "Tamil Nadu, India", "Grey", "Low", "Excellent draught capacity, hardy breed"),
    ("Kankrej", "Gujarat, India", "Silver grey", "1400–1800 L/year", "Good milk yield and heat resistance"),
    ("Kasargod", "Kerala, India", "Black or dark brown", "Low", "Small indigenous draught breed"),
    ("Kenkatha", "Uttar Pradesh, India", "Grey or light black", "Low", "Draught breed for heavy work"),
    ("Kherigarh", "Uttar Pradesh, India", "White or grey", "Low", "Light draught animal, good endurance"),
    ("Khillari", "Maharashtra, India", "Grey to white", "Low", "Adapted to dry climate, draught type"),
    ("Krishna_Valley", "Karnataka, India", "Grey white", "1000–1500 L/year", "Dual-purpose, known for hardiness"),
    ("Malnad_gidda", "Karnataka, India", "Dark brown or black", "Low", "Small-sized, disease-resistant"),
    ("Mehsana", "Gujarat, India", "Black", "1200–1800 L/year", "Buffalo breed with high butterfat milk"),
    ("Murrah", "Haryana, India", "Jet black", "1800–2500 L/year", "Best buffalo breed for milk yield"),
    ("Nagori", "Rajasthan, India", "White", "Low", "Fast and powerful draught animal"),
    ("Nagpuri", "Maharashtra, India", "Black", "1000–1500 L/year", "Dual-purpose buffalo breed"),
    ("Nili_Ravi", "Punjab (India & Pakistan)", "Black", "2000–3000 L/year", "High milk yield, popular dairy buffalo"),
    ("Nimari", "Madhya Pradesh, India", "Reddish brown", "Low", "Good draught capacity"),
    ("Ongole", "Andhra Pradesh, India", "White or grey", "1200–1500 L/year", "Exported worldwide, strong build"),
    ("Pulikulam", "Tamil Nadu, India", "Grey", "Low", "Used for Jallikattu and draught work"),
    ("Rathi", "Rajasthan, India", "Brown with white patches", "1500–2000 L/year", "Excellent dual-purpose breed"),
    ("Red_Dane", "Denmark", "Reddish brown", "6000+ L/year", "High milk yield, docile temperament"),
    ("Red_Sindhi", "Sindh, Pakistan", "Red or brown", "1500–2000 L/year", "High milk fat and heat tolerance"),
    ("Sahiwal", "Punjab, India", "Reddish dun", "1800–2200 L/year", "Best Indian dairy breed, calm and disease-resistant"),
    ("Surti", "Gujarat, India", "Light grey", "1200–1800 L/year", "Buffalo breed with high milk fat"),
    ("Tharparkar", "Rajasthan, India", "White or grey", "1500–2000 L/year", "Good milk yield, heat and drought tolerant"),
    ("Toda", "Nilgiris, Tamil Nadu", "Brown or fawn", "Low", "Reared by Toda tribes, small population"),
    ("Umblachery", "Tamil Nadu, India", "Grey", "Low to medium", "Dual-purpose, good for draught work"),
    ("Vechur", "Kerala, India", "Light brown", "450–500 L/year", "World’s smallest cattle breed, high fat milk")
]

cursor.executemany("""
INSERT OR IGNORE INTO breeds (name, origin, color, milk_yield, characteristics)
VALUES (?, ?, ?, ?, ?)
""", breeds_data)

conn.commit()
conn.close()

print("cows.db created successfully with breed data!")
