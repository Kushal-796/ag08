import pandas as pd
import json

# Load the CSV
df = pd.read_csv("crop_data.csv")

# Group districts by state
state_districts = {}
for state, group in df.groupby('State_Name'):
    districts = sorted(group['District_Name'].dropna().unique())
    state_districts[state] = districts

# Save to JSON
with open("state_districts.json", "w") as f:
    json.dump(state_districts, f, indent=2)

print("Saved state_districts.json with state to districts mapping.")