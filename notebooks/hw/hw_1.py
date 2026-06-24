# IMPORT PANDAS TOOLS
import pandas as pd
import requests
import io

democracy_gdp_filepath = "../../data/hw/democracy_gdp.csv"  
democracy_gdp = pd.read_csv(democracy_gdp_filepath)
democracy_gdp.head(4)

ghg_file_path = "../../data/hw/democracy_gdp.csv"           
ghg = pd.read_csv(ghg_file_path)

print(ghg.tail(6))

def download_worldbank(indicator, countries, date_start, date_end):
    url_base = 'https://api.worldbank.org/v2/'
    country_codes = ';'.join(countries)
    url = f'{url_base}country/{country_codes}/indicator/{indicator}?date={date_start}:{date_end}&per_page=30000'
    response = requests.get(url)
    df = pd.read_xml(io.BytesIO(response.content))
    return df

gdp_data = download_worldbank(
    indicator='NY.GDP.PCAP.CD',
    date_start=1990,
    date_end=2026,
    countries=["USA", "CAN", "MEX", "GBR", "FRA", "DEU", "JPN", "AUS"],
)

print(gdp_data.head(6))

population_indicator = "SP.POP.TOTL"   # World Bank code for total population

country_list = [
    "USA",  # United States of America
    "CAN",  # Canada
    "MEX",  # Mexico
    "GBR",  # United Kingdom of Great Britain and Northern Ireland
    "FRA",  # France
    "DEU",  # Germany (Deutschland)
    "JPN",  # Japan
    "AUS"   # Australia
]

pop_data = download_worldbank(
    indicator=population_indicator,
    countries=country_list,
    date_start=1990,
    date_end=2026
)

print(pop_data.tail(6))

# Selecting a column
democracy_gdp['country']

# Unique values in a column
democracy_gdp['country'].unique()

# Filter using query
democracy_gdp.query("country == 'AFG'")

# Filter with multiple conditions
democracy_gdp[(democracy_gdp['country'] == 'AFG') & (democracy_gdp['year'] > 2000)]

# Filter multiple and show unique countries
print(democracy_gdp.columns.tolist())
asia_df = democracy_gdp[democracy_gdp['country'].str.contains('Asia', case=False, na=False)]
asia_df['country'].unique()

# Filter rows and select columns
democracy_gdp.loc[democracy_gdp['country'] == 'AFG', ['country', 'year', 'NGDPDPC']]

# Before rename
print(democracy_gdp.head(2))

# Rename using rename with axis parameter
democracy_gdp.rename({'country': 'Country', 'NGDPDPC': 'GDP_per_capita'}, axis='columns')
