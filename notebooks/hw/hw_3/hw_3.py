import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pycountry

# You can use example datasets here docs/example_datasets.md

# 1. Print environment details (working directory and local folder contents)
# If you read csv/parquet etc use relative paths such as data/hw/hw_3/democracy_gdp.csv
print("Current working directory:", os.getcwd())
print("Files in project directory:", os.listdir())


# 2. Acquire and load your primary dataset
df = pd.read_csv("../../../data/hw/democracy_gdp.csv")
print(df.head())


# 1. Inspect data structure using pandas methods

print(df.info())
print(df.head())

# 2. Rename columns and select a clean subset
column_rename_map = {
    'v2x_frassoc_thick': 'freedom_association',
    'NGDPDPC': 'gdp_per_capita',
    'NGDP_RPCH': 'gdp_growth',
    'v2x_pubcorr': 'public_corruption',
    'v2xnp_regcorr': 'regime_corruption',
    'v2xel_frefair': 'free_fair_elections',
    'v2x_freexp': 'freedom_expression',
    'v2elembcap': 'electoral_capacity'
}

df = df.rename(columns=column_rename_map)

df = df[['country', 'year', 'gdp_per_capita', 'gdp_growth', 
         'freedom_association', 'free_fair_elections', 
         'freedom_expression', 'public_corruption']]

df = df[~df['country'].str.contains(r'\(avg\)', na=False)]

def iso3_to_name(code):
    try:
        return pycountry.countries.get(alpha_3=code).name
    except AttributeError:
        return None

df['country'] = df['country'].apply(iso3_to_name)
df = df.dropna(subset=['country'])

country_overrides = {
    'Korea, Republic of': 'Korea, South',
    'Taiwan, Province of China': 'Taiwan',
    'Hong Kong': 'Hong Kong',
    # add more as you discover them after running the overlap check again
}
df['country'] = df['country'].replace(country_overrides)

print(df.head())

# 3. Filter row or column filters if necessary
df = df[df['year'] >= 2000] 
print(df.shape)

# 4. Handle missing values (NaNs) if necessary
print("Missing values before cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("Missing values after cleaning:")
print(df.isnull().sum())
print(df.head())

# 2. Rename columns and select a clean subset

column_rename_map = {
    'v2x_frassoc_thick': 'freedom_association',
    'NGDPDPC': 'gdp_per_capita',
    'NGDP_RPCH': 'gdp_growth',
    'v2x_pubcorr': 'public_corruption',
    'v2xnp_regcorr': 'regime_corruption',
    'v2xel_frefair': 'free_fair_elections',
    'v2x_freexp': 'freedom_expression',
    'v2elembcap': 'electoral_capacity'
}

df = df.rename(columns=column_rename_map)

df = df[['country', 'year', 'gdp_per_capita', 'gdp_growth', 
         'freedom_association', 'free_fair_elections', 
         'freedom_expression', 'public_corruption']]

print(df.head())

# 3. Filter row or column filters if necessary
df = df[df['year'] >= 2000] 
print(df.shape)

# 4. Handle missing values (NaNs) if necessary
print("Missing values before cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("Missing values after cleaning:")
print(df.isnull().sum())
print(df.head())

# 1. Load and clean the secondary dataset
df_secondary = pd.read_csv("../../../data/hw/hw_3/eichengreen_1.csv")

print(df_secondary.info())
print(df_secondary.head())

secondary_rename_map = {
    'date': 'year',
}

df_secondary = df_secondary.rename(columns=secondary_rename_map)
df_secondary = df_secondary.dropna()
print(df_secondary.head())


print("cheking --------------")
print(df['year'].min(), df['year'].max())
print(df_secondary['year'].min(), df_secondary['year'].max())


# 2. Merge your datasets on a common key

df_merged = pd.merge(df, df_secondary, on=['country', 'year'], how='inner')   # ← FIRST merge, on dirty/unmatched strings

print("Merged dataframe shape:", df_merged.shape)
print(df_merged.columns.tolist())   
print(df_merged.head())

print(sorted(df['country'].unique())[:20])
print(sorted(df_secondary['country'].unique())[:20])

# Check overlap
'''
...
common = set(df['country'].unique()) & set(df_secondary['country'].unique())
print("Number of overlapping country names:", len(common))
...
'''
common = set(df['country'].unique()) & set(df_secondary['country'].unique())
print(len(common))
print(common)

df['country'] = df['country'].str.strip()
df_secondary['country'] = df_secondary['country'].str.strip()

df_merged = pd.merge(df, df_secondary, on=['country', 'year'], how='inner')   # ← SECOND merge, this overwrites df_merged
print(df_merged.shape)

print(" results from last requests")

print(df_merged.shape)
print(common)

print(sorted(df['country'].unique()))

print(sorted(df_secondary['country'].unique()))

print(df['country'].dtype, df_secondary['country'].dtype)
print(repr(df['country'].iloc[0]))
print(repr(df_secondary['country'].iloc[0]))


# Apply groupby aggregation or implement one of the backup grouping strategies
# I grouped them by year to see how variables trend over time globally

year_avg = df_merged.groupby('year').agg(
    avg_gdp_per_capita=('gdp_per_capita', 'mean'),
    avg_gdp_growth=('gdp_growth', 'mean'),
    avg_freedom_association=('freedom_association', 'mean'),
    avg_public_corruption=('public_corruption', 'mean')
).reset_index()

print(year_avg)



# Plot 1: Customized bar chart or line chart of aggregated subgroups
# Divided by GDP 
plt.figure(figsize=(10, 6))
plt.plot(year_avg['year'], year_avg['avg_gdp_per_capita'], marker='o', color='steelblue')
plt.xlabel('Year')
plt.ylabel('Average GDP per Capita')
plt.title('Average GDP per Capita Over Time (2000+)')
plt.tight_layout()
plt.show()



# Defining predictors and outcome
predictors = ['freedom_association', 'free_fair_elections', 'freedom_expression', 'public_corruption']
X = df_merged[predictors]
y = df_merged['gdp_growth']

X = sm.add_constant(X)

model = sm.OLS(y, X, missing='drop').fit()
print(model.summary())


# Plot 2: Customized scatter plot of primary policy variables
plt.figure(figsize=(8, 6))
plt.scatter(df_merged['free_fair_elections'], df_merged['gdp_growth'],
            alpha=0.5, color='darkorange', edgecolor='black', linewidth=0.3)
plt.xlabel('Free and Fair Elections Index')
plt.ylabel('GDP Growth (%)')
plt.title('Free and Fair Elections vs. GDP Growth')
plt.tight_layout()
plt.show()


# Fit a simple linear regression and overlay the line of best fit
x = df_merged['free_fair_elections']
y = df_merged['gdp_growth']

# Fit using numpy's polyfit (degree 1 = linear)
slope, intercept = np.polyfit(x, y, 1)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, color='darkorange', edgecolor='black', linewidth=0.3, label='Observed data')
plt.plot(line_x, line_y, color='navy', linewidth=2, label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')
plt.xlabel('Free and Fair Elections Index')
plt.ylabel('GDP Growth (%)')
plt.title('Free and Fair Elections vs. GDP Growth with Linear Fit')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")


