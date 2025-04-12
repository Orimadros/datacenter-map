import geopandas as gpd
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading data...")
census_tracts = gpd.read_file('data/model_input/brasil/sao_paulo_census_tracts_full.geojson')
data_centers = gpd.read_file('data/processed/brasil/data_centers.geojson')

print(f"Total census tracts in São Paulo state: {len(census_tracts)}")
print(f"Total data centers in Brazil: {len(data_centers)}")

# Create a dictionary to map census tract IDs to municipality names
# First 7 digits of CD_SETOR represent the municipality code (CD_MUN)
print("Creating municipality mapping...")
census_tracts['CD_MUN'] = census_tracts['CD_SETOR'].astype(str).str[:7]

# Load IBGE municipality codes from census tracts
municipalities = {}
mun_counts = Counter(census_tracts['CD_MUN'])
print(f"Found {len(mun_counts)} municipalities in São Paulo state")

# Map of IBGE codes to municipality names (hardcoded for São Paulo state's main cities)
mun_names = {
    '3550308': 'São Paulo',
    '3509502': 'Campinas',
    '3525904': 'Jundiaí',
    '3552205': 'Sumaré',
    '3537107': 'Paulínia',
    '3519071': 'Hortolândia',
    '3556453': 'Vinhedo',
    '3529401': 'Osasco',
    '3506003': 'Barueri',
    '3547304': 'Santana de Parnaíba',
    '3539103': 'Piracicaba',
    '3548708': 'São Bernardo do Campo',
    '3548500': 'São André',
    '3548807': 'São Caetano do Sul',
    '3510609': 'Carapicuíba',
    '3534401': 'Nova Odessa',
    '3526209': 'Limeira',
    '3512803': 'Cotia',
    '3513801': 'Diadema',
    '3505708': 'Bauru',
    '3543303': 'Ribeirão Preto',
    '3549805': 'São José dos Campos',
    '3550704': 'São Vicente',
    '3551009': 'Sorocaba',
    '3513009': 'Cubatão',
    '3541000': 'Praia Grande',
    '3518800': 'Guarulhos',
    '3552502': 'Suzano',
    '3522208': 'Indaiatuba',
    '3520509': 'Guarujá',
}

# Ensure same CRS for data centers and census tracts
if data_centers.crs != census_tracts.crs:
    print(f"Converting data centers from {data_centers.crs} to {census_tracts.crs}")
    data_centers = data_centers.to_crs(census_tracts.crs)

# Create a dissolved boundary for São Paulo state
print("Creating São Paulo state boundary...")
sp_state_boundary = census_tracts.unary_union

# Filter data centers to those within the São Paulo state boundary
sp_data_centers = data_centers[data_centers.geometry.within(sp_state_boundary)].copy()
print(f"Data centers in São Paulo state: {len(sp_data_centers)}")

# Create municipality polygons by dissolving census tracts by municipality code
print("Creating municipality boundaries...")
municipalities_gdf = census_tracts.dissolve(by='CD_MUN').reset_index()

# Perform spatial join to find which municipality each data center is in
print("Performing spatial join to find data center municipalities...")
joined = gpd.sjoin(sp_data_centers, municipalities_gdf, how='left', predicate='within')

# Extract municipality codes and map to names
joined['municipality_code'] = joined['CD_MUN']
joined['municipality'] = joined['municipality_code'].map(mun_names)

# Create frequency table
municipality_counts = Counter(joined['municipality'])
frequency_table = pd.DataFrame.from_dict(municipality_counts, orient='index', columns=['count'])
frequency_table = frequency_table.reset_index().rename(columns={'index': 'municipality'})
frequency_table = frequency_table.sort_values('count', ascending=False)

print("\nFrequency table of data centers by municipality:")
print(frequency_table)

# Save the results to CSV
output_path = 'outputs/sao_paulo_datacenter_municipalities.csv'
frequency_table.to_csv(output_path, index=False)
print(f"Saved frequency table to {output_path}")

# Create a bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='municipality', data=frequency_table.head(15))
plt.title('Top 15 Municipalities by Data Center Count in São Paulo State')
plt.xlabel('Number of Data Centers')
plt.ylabel('Municipality')
plt.tight_layout()

# Save the plot
plot_path = 'outputs/figures/datacenter_municipality_distribution.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_path}")

# Print data centers by municipality for the top 5 municipalities
top_municipalities = frequency_table.head(5)['municipality'].tolist()
print("\nData centers in top 5 municipalities:")
for mun in top_municipalities:
    mun_dcs = joined[joined['municipality'] == mun]
    print(f"\n{mun} ({len(mun_dcs)} data centers):")
    if 'name' in mun_dcs.columns:
        for name in mun_dcs['name']:
            print(f"- {name}")
    elif 'title' in mun_dcs.columns:
        for title in mun_dcs['title']:
            print(f"- {title}")

# Create a simple map showing municipalities and data centers
print("\nCreating map of data centers by municipality...")
try:
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot all municipalities with light gray
    municipalities_gdf.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.5)
    
    # Plot top 5 municipalities with a color palette
    top_mun_codes = joined[joined['municipality'].isin(top_municipalities)]['municipality_code'].unique()
    if len(top_mun_codes) > 0:
        top_municipalities_gdf = municipalities_gdf[municipalities_gdf['CD_MUN'].isin(top_mun_codes)]
        top_municipalities_gdf.plot(ax=ax, cmap='Blues', edgecolor='blue', linewidth=1, alpha=0.7)
    
    # Add data centers
    sp_data_centers.plot(ax=ax, marker='o', color='red', markersize=30, edgecolor='black')
    
    plt.title("Data Centers in São Paulo State by Municipality")
    plt.axis('off')
    
    # Save the map
    map_path = 'outputs/figures/datacenter_municipality_map.png'
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    print(f"Saved map to {map_path}")
except Exception as e:
    print(f"Error creating map: {e}") 