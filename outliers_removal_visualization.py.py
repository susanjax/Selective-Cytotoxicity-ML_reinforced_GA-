import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import descriptor_addition

original_data = pd.read_csv('data/new_combined.csv')
df_with_descriptors = descriptor_addition(original_data)
# function for outlier_removal


df = df_with_descriptors.drop(['CID', 'Canonical_smiles'], axis=1)
df['Valance_electron'] = df['Valance_electron'].astype(float)
df['time (hr)'] = df['time (hr)'].astype(float)

df_imp = df[['time (hr)',
       'concentration (ug/ml)', 'viability (%)', 'Hydrodynamic diameter (nm)',
       'Zeta potential (mV)','mcd', 'electronegativity', 'rox', 'radii',]]

# violin plots of important numerical parameters
fig2, ax_ = plt.subplots(3, 3, figsize=(12, 11))
ax = []
for i in ax_:
    ax += i.tolist()
for number, column in enumerate(df_imp.columns):
    sns.violinplot(data=df_imp, x=column, ax=ax[number])
fig2.suptitle("Violin plots for columns in db")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)
plt.show()

""" remove outliers
removing 1% of the data from the end where data are widely spread
"""
# check concentration distribution and remove outlier
sns.displot(data=df, x='concentration (ug/ml)', kind='hist')
conc = df["concentration (ug/ml)"].quantile(0.99)
# print(conc)
df2 = df[df['concentration (ug/ml)']<1001]

# check viability distribution and remove outlier
sns.displot(data=df2, x='viability (%)', kind='hist')
viab = df["viability (%)"].quantile(0.99)
# print(viab)
df3 = df2[df2['viability (%)']< 126.185]

# check hydrodynamic diameter distribution and remove outlier
sns.displot(data=df3, x='Hydrodynamic diameter (nm)', kind='hist')


hd = df["Hydrodynamic diameter (nm)"].quantile(0.99)
# print(hd)
df4 = df3[df3['Hydrodynamic diameter (nm)']< 600]

# check zeta potential distribution and remove outlier
sns.displot(data=df4, x='Zeta potential (mV)', kind='hist')
zeta = df["Zeta potential (mV)"].quantile(0.01)
# print(zeta)
df5 = df4[df4['Zeta potential (mV)']>-63.5]

df_imp = df5[['time (hr)',
       'concentration (ug/ml)', 'viability (%)', 'Hydrodynamic diameter (nm)',
       'Zeta potential (mV)', 'mcd', 'electronegativity', 'rox', 'radii',]]

# violin plots
fig2, ax_ = plt.subplots(3, 3, figsize=(12, 11))
ax = []
for i in ax_:
    ax += i.tolist()
for number, column in enumerate(df_imp.columns):
    sns.violinplot(data=df_imp, x=column, ax=ax[number])
fig2.suptitle("Violin plots for columns in db")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.6)
plt.show()


"""##remove low variance values using variance threshold """

from sklearn.feature_selection import VarianceThreshold
def variance_threshold(df,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]

"""separating categorical and numerical values"""

# df5ss.info()

dff5=df5.select_dtypes(include=['float64'])
dff5o = df5.select_dtypes(include=['object'])

dff5.columns

df6 = variance_threshold(dff5, 0)
df6

sns.heatmap(df6.corr())
sns.set(rc={'figure.figsize':(20,15)})

# df7 = variance_threshold(dff5, 0.8*(1-0.8))
# df7

# df7.corr()

# sns.heatmap(df7.corr())
# sns.set(rc={'figure.figsize':(20,15)})

"""##remove highly corelated columns"""

import numpy as np
def corr(df, val):
  corr_matrix = df.corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
  to_drop = [column for column in upper.columns if any(upper[column] > val)]
  return df.drop(to_drop, axis=1, inplace=True)

corr(df6, 0.90)
df6
sns.heatmap(df6.corr())
sns.set(rc={'figure.figsize':(20,15)})

"""#dataset after feature removal
merge numerical and catergorical values
"""

df6_all = pd.merge(dff5o, df6, left_index=True, right_index=True)
preprocessed = df6_all.copy()
preprocessed = preprocessed.reset_index(drop=True)

# preprocessed.to_csv('preprocessed_data.csv')

print(preprocessed)



def outlier_remove(data_with_descriptor):
    df = df_with_descriptors.drop(['CID', 'Canonical_smiles'], axis=1)
    df['Valance_electron'] = df['Valance_electron'].astype(float)
    df['time (hr)'] = df['time (hr)'].astype(float)
    df2 = df[df['concentration (ug/ml)'] < 1001]
    df3 = df2[df2['viability (%)'] < 126.185]
    df4 = df3[df3['Hydrodynamic diameter (nm)'] < 600]
    df5 = df4[df4['Zeta potential (mV)'] > -63.5]
    return df5
# print(outlier_remove(descriptor_addition(original_data)))

def remove_correlation(data_with_outlier_removed):
    df5 = data_with_outlier_removed
    dff5 = df5.select_dtypes(include=['float64'])
    dff5o = df5.select_dtypes(include=['object'])
    df6 = variance_threshold(dff5, 0)
    corr(df6, 0.90)
    df6_all = pd.merge(dff5o, df6, left_index=True, right_index=True)
    return df6_all

preprocessed = remove_correlation(outlier_remove(descriptor_addition(original_data)))
original_only = preprocessed[preprocessed['source']=='original']
additional_only = preprocessed[preprocessed['source']=='additional']
test = preprocessed[preprocessed['source']=='test']
positive = preprocessed[preprocessed['source']=='positive']
preprocessed = preprocessed.drop(['index', 'source'],axis=1)

'''Categorical data visualization'''

'''material'''
mat = preprocessed.groupby('material').count()
mat_s = mat.sort_values(by= 'cell line', ascending=False)



# Filter out the rows in the counts dataframe where the 'Occurrence' column is less than 1%
# counts_less_than_1 = mat_s.where(mat_s["cell line"] / counts["cell line"].sum() * 100 < 1).dropna()

counts_less_than_1 = mat_s.where(mat_s["cell line"] / mat_s["cell line"].sum() * 100 < 1).dropna()
counts_more_than_1 = mat_s.where(mat_s["cell line"] / mat_s["cell line"].sum() * 100 >= 1).dropna()

# sum the less than 1% rows and call it 'other'
counts_less_than_1 = counts_less_than_1.sum()
counts_less_than_1.name = 'other'

# append 'other' row to the main dataframe
counts_more_than_1 = counts_more_than_1.append(counts_less_than_1)

# Plot the pie chart
sns.set(rc={'figure.figsize':(15,15)})
counts_more_than_1.plot.pie(y='cell line', labels=counts_more_than_1.index, autopct='%1.1f%%')
plt.title("Prevalence of different nanomaterial in dataset")
plt.show()

'''cell line'''
cell = preprocessed.groupby('cell line').count()
cell_s = cell.sort_values(by= 'test', ascending=False)
# Filter out the rows in the counts dataframe where the 'Occurrence' column is less than 1%
# counts_less_than_1 = mat_s.where(mat_s["cell line"] / counts["cell line"].sum() * 100 < 1).dropna()
counts_less_than_1 = cell_s.where(cell_s["test"] / cell_s["test"].sum() * 100 < 1.5).dropna()
counts_more_than_1 = cell_s.where(cell_s["test"] / cell_s["test"].sum() * 100 >= 1.5).dropna()

# sum the less than 1% rows and call it 'other'
counts_less_than_1 = counts_less_than_1.sum()
counts_less_than_1.name = 'other'

# append 'other' row to the main dataframe
counts_more_than_1 = counts_more_than_1.append(counts_less_than_1)

# Plot the pie chart
counts_more_than_1.plot.pie(y='test', labels=counts_more_than_1.index, autopct='%1.1f%%')
plt.legend(loc='lower right')
plt.title("Prevalence of different cell line in dataset")
plt.show()

'''test'''
test = preprocessed.groupby('test').count()
test_s = test.sort_values(by= 'material',
ascending=False)
# Filter out the rows in the counts dataframe where the 'Occurrence' column is less than 1%
# counts_less_than_1 = mat_s.where(mat_s["cell line"] / counts["cell line"].sum() * 100 < 1).dropna()
counts_less_than_1 = test_s.where(test_s["material"] / test_s["material"].sum() * 100 < 0.5).dropna()
counts_more_than_1 = test_s.where(test_s["material"] / test_s["material"].sum() * 100 >= 0.5).dropna()

# sum the less than 1% rows and call it 'other'
counts_less_than_1 = counts_less_than_1.sum()
counts_less_than_1.name = 'other'

# append 'other' row to the main dataframe
counts_more_than_1 = counts_more_than_1.append(counts_less_than_1)

# Plot the pie chart
counts_more_than_1.plot.pie(y='material', labels=counts_more_than_1.index, autopct='%1.1f%%')
plt.title("Prevalence of different test type in dataset")
plt.show()