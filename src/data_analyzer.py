import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from data_loader import load_data

os.makedirs("../plots", exist_ok=True)

print("Files loaded:")
df = load_data()

print("\nShape (rows, columns): ", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=60, color="#2a6f97")
plt.title("Price distribution")
plt.xlabel("Price (£)")
plt.ylabel("Number of cars")
plt.tight_layout()
plt.savefig("../plots/1_price_distribution.png")
plt.close()

price_by_year = df[(df["year"] >= 2000) & (df["year"] <= 2020)].groupby("year")["price"].mean()
plt.figure(figsize=(8, 5))
plt.plot(price_by_year.index, price_by_year.values, marker="o", color="#2a6f97")
plt.title("Average price by year")
plt.xlabel("Manufacturing year")
plt.ylabel("Average price (£)")
plt.tight_layout()
plt.savefig("../plots/2_price_by_year.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["mileage"] * 1.609, df["price"], s=2, alpha=0.2, color="#2a6f97")
plt.title("Price by mileage")
plt.xlabel("Mileage (km)")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("../plots/3_price_vs_mileage.png")
plt.close()

price_by_brand = df.groupby("brand")["price"].mean().sort_values()
plt.figure(figsize=(8, 5))
plt.barh(price_by_brand.index, price_by_brand.values, color="#2a6f97")
plt.title("Average price by brand")
plt.xlabel("Average price (£)")
plt.tight_layout()
plt.savefig("../plots/4_price_by_brand.png")
plt.close()

print("\nPlots saved in the plots/ folder")
