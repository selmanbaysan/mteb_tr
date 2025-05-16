import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the dataset
file_path = "benchmark_tables/MTEB(Turkish)_summary.csv"
df = pd.read_csv(file_path)

# Convert numeric columns from string to float (if needed)
numeric_columns = [
    "Mean (Task)", "Mean (TaskType)", "Bitext Mining","Classification",
    "Clustering", "Pair Classification", "Retrieval", "STS"
]
df[numeric_columns] = df[numeric_columns].astype(float)

# get the model name only inside [ ] brackets
def filter_model_name(model_name):
    model_name_match = re.search(r"\[(.*?)\]", model_name)
    if model_name_match:
        return model_name_match.group(1)
    return model_name


df["Model"] = df["Model"].apply(filter_model_name)

# Sort by Rank
df = df.sort_values(by="Rank (Borda)")

# 📊 Visualization 1: Benchmark Table
plt.figure(figsize=(12, 6))
sns.heatmap(df[numeric_columns].set_index(df["Model"]), annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.title("Turkish MTEB Benchmark Performance")
plt.ylabel("Model")
plt.xlabel("Task")
plt.xticks(rotation=45)
plt.gca().set_position([0.3, 0.1, 0.6, 0.8])  # Shift the plot to the right
plt.savefig("benchmark_tables/benchmark_table.png", bbox_inches="tight")

# 📊 Visualization 2: Bar Plot for Overall Performance (Mean Task Score)
plt.figure(figsize=(12, 6))
sns.barplot(y=df["Model"], x=df["Mean (Task)"], palette="viridis", edgecolor="black")
plt.title("Overall Mean Task Performance")
plt.xlabel("Mean (Task) Score")
plt.ylabel("Model")
plt.xlim(df["Mean (Task)"].min() - 2, df["Mean (Task)"].max() + 2)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.gca().set_position([0.3, 0.1, 0.6, 0.8])  # Shift the plot to the right
plt.savefig("benchmark_tables/overall_performance.png", bbox_inches="tight")


# 📊 Visualization 3: Task-Specific Performance Comparison
plt.figure(figsize=(14, 7))
df_melted = df.melt(id_vars=["Model"], value_vars=numeric_columns, var_name="Task", value_name="Score")
sns.boxplot(x="Task", y="Score", data=df_melted, palette="Set2", hue="Task", legend=False)
plt.xticks(rotation=30)
plt.title("Performance Distribution Across Tasks")
plt.xlabel("Task")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.gca().set_position([0.3, 0.1, 0.6, 0.8])  # Shift the plot to the right
plt.savefig("benchmark_tables/task_performance_comparison.png", bbox_inches="tight")