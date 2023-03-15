import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
df1 = pd.read_csv('other data/Model_comparision_test_raw.csv')
df2 = pd.read_csv('other data/Model_comparision_test.csv')

# Remove last 5 rows
# df1 = df1.iloc[:-5, :]
# df2 = df2.iloc[:-5, :]

# Add a column to each data frame indicating the source
df1['Source'] = 'Raw Data'
df2['Source'] = 'Processed Data'

# Concatenate the data frames
df = pd.concat([df2, df1])

# Filter out rows with negative R-Squared values
df = df[df['Adjusted R-Squared'] >= 0]

# Set plot style and font sizes
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
sns.set(font="Times New Roman")

# Define colors for raw and processed data
colors = {"Raw Data": "#3182bd", "Processed Data": "#fd8d3c"}

# Create horizontal bar plot
plt.figure(figsize=(8, 16))
ax = sns.barplot(x="Adjusted R-Squared", y="Model", hue="Source", data=df, palette=colors)

# Set axis labels and title
ax.set_xlabel("Adjusted R-Squared", fontsize=16, fontname="Times New Roman")
ax.set_ylabel("Model", fontsize=16, fontname="Times New Roman")
plt.title("Comparison of Regression Models", fontsize=18, fontweight="bold", fontname="Times New Roman")

# Set tick label font sizes and font style
plt.xticks(fontsize=14, fontname="Times New Roman")
plt.yticks(fontsize=14, fontname="Times New Roman")

# Set legend font size and location
plt.legend(fontsize=14, loc="lower right")

# Save plot to file
plt.tight_layout()
plt.show()
# plt.savefig("models_r_squared_comparison.png", dpi=300)
