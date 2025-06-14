import pandas as pd
import re

# Read the file content
with open('Parallel/oberservations.txt', 'r') as file:
    content = file.read()

# Define a regex pattern for matching the records in the file
pattern = re.compile(
    r"Schedule:\s*(.*?)\n" +
    r"Num threads:\s*(.*?)\n" +
    r"Type of parallelism:\s*(.*?)\n" +
    r"With/without barriers:\s*(.*?)\n" +
    r"Recorded execution time \(in seconds\):\s*(.*?)\n" +
    r"Best path length:\s*(.*?)\n" +
    r"Dataset:\s*(.*?)\n",
    re.IGNORECASE
)

# Find all matches in the file content
matches = pattern.findall(content)

# Create a DataFrame from the captured groups
df = pd.DataFrame(matches, columns=[
    "Schedule",
    "Num threads",
    "Type of parallelism",
    "With/without barriers",
    "Execution time (s)",
    "Best path length",
    "Dataset"
])

# Replace Schedule with "N/A" where Type of parallelism is "Task"
df.loc[df["Type of parallelism"].str.strip().str.lower() == "task", "Schedule"] = "N/A"

# Save the DataFrame to a CSV file in the /mnt/data/ directory
csv_filename = 'Parallel/observations.csv'
df.to_csv(csv_filename, index=False)

print(f"CSV file has been saved as {csv_filename}")
