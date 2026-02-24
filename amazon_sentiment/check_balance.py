from datasets import load_dataset
import numpy as np
from collections import Counter

# Učitavanje skupa (samo prvih N primera ako je prevelik)
print("Učitavam amazon_polarity dataset...")
ds = load_dataset("mteb/amazon_polarity")

# Analiza trening skupa (možete i test)
train_labels = ds["train"]["label"]  # 0 = negative, 1 = positive

# Ograniči na prvih 10.000 primera ako želite brže
sample_size = min(10000, len(train_labels))
sample_labels = train_labels[:sample_size]

# Izračunaj distribuciju
counts = Counter(sample_labels)
total = sum(counts.values())

print(f"\nAnaliza {sample_size} nasumičnih primera iz trening skupa:")
print(f"Negativnih (0): {counts[0]} ({counts[0]/total*100:.1f}%)")
print(f"Pozitivnih (1): {counts[1]} ({counts[1]/total*100:.1f}%)")
print(f"Odnos: {counts[0]/counts[1]:.3f} : 1")

# Dodatno: provera celog skupa ako nije prevelik
if len(train_labels) <= 100000:
    full_counts = Counter(train_labels)
    print(f"\nCeo trening skup ({len(train_labels)} primera):")
    print(f"Negativnih: {full_counts[0]} ({full_counts[0]/len(train_labels)*100:.1f}%)")
    print(f"Pozitivnih: {full_counts[1]} ({full_counts[1]/len(train_labels)*100:.1f}%)")