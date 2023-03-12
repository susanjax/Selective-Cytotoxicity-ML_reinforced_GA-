
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a file or create a DataFrame
df = pd.read_csv('pop_size_all.csv')



# Set the style and color palette
sns.set_style('whitegrid')
sns.set_palette(['#e74c3c', '#3498db', '#2ecc71','#e74c3c', '#3498db', '#2ecc71',])


# Set the size of the figure
plt.figure(figsize=(12, 10))

# Create six line plots using Seaborn
sns.lineplot(data=df, x='generations', y='Population size = 100', label='Pop. Size = 100', ci=None, estimator='mean')
sns.lineplot(data=df, x='generations', y='Population size = 50', label='Pop. Size = 50', ci=None, estimator='mean')
sns.lineplot(data=df, x='generations', y='Population size = 10', label='Pop. Size = 10', ci=None, estimator='mean')
sns.lineplot(data=df, x='generations', y='mean of pop 10', label='Mean Pop. Size = 10', alpha=0.5)
sns.lineplot(data=df, x='generations', y='mean_of pop 50', label='Mean Pop. Size = 50', alpha=0.5)
sns.lineplot(data=df, x='generations', y='mean_of pop 100', label='Mean Pop. Size = 100', alpha=0.5)

# Set the title and axis labels
plt.title('Fitness Score of Different Populations', fontsize=16, color='black')
plt.xlabel('Generation Numbers', fontsize=14, color='black')
plt.ylabel('Fitness Score', fontsize=14, color='black')

# Set the thickness and color of x and y axis
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# Set the thickness and color of ticks on the x and y axis
plt.gca().tick_params(axis='x', width=2, color='black')
plt.gca().tick_params(axis='y', width=2, color='black')

plt.gca().set_xlim([0, 100])
plt.gca().set_ylim([0, 35])

# Create two separate legends
max_fitness_legend = plt.legend(title='Max Fitness', labels=['Pop. Size = 100', 'Pop. Size = 50', 'Pop. Size = 10'], bbox_to_anchor=(1.02, 1), loc='upper left')
mean_fitness_legend = plt.legend(title='Mean Fitness', labels=['Pop. Size = 100', 'Pop. Size = 50', 'Pop. Size = 10'], bbox_to_anchor=(1.02, 0.6), loc='upper left')

# Add both legends to the plot
plt.gca().add_artist(max_fitness_legend)
plt.gca().add_artist(mean_fitness_legend)

# Show the plot
plt.show()

