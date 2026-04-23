import io, sys, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.data_loader import load_and_clean

df = load_and_clean()

print()
print("=== Sample datetimes (should have real HH:MM, NOT 12:00:00) ===")
print(df[["datetime", "hour"]].head(20).to_string())

print()
print("=== Hour distribution (should span 0-23) ===")
print(df["hour"].value_counts().sort_index().to_string())

print()
print("Unique hours:", sorted(df["hour"].unique()))
print("Min datetime:", df["datetime"].min())
print("Max datetime:", df["datetime"].max())
print("All rows at midnight?:", (df["datetime"].dt.hour == 0).all())
