import pandas as pd

df = pd.read_csv("RECOMMENDATION.csv", engine="python")

# Keep only required columns
df = df.iloc[:, :6]

df.columns = ['skin_tone','undertone','occasion','weather','colors','avoid_colors']

# Clean spaces
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Ensure avoid_colors in quotes format (optional consistency)
df['avoid_colors'] = df['avoid_colors'].astype(str)

# Drop broken rows
df = df.dropna()

# Save clean file
df.to_csv("clean_recommendation.csv", index=False)

print("Cleaned Successfully ✅")