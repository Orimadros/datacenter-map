import geopandas as gpd

# Load data
print("Loading data...")
census_tracts = gpd.read_file('data/model_input/brasil/sao_paulo_census_tracts_full.geojson')
data_centers = gpd.read_file('data/processed/brasil/data_centers.geojson')

print(f"Total census tracts in São Paulo state: {len(census_tracts)}")
print(f"Total data centers in Brazil: {len(data_centers)}")

# Create a dissolved boundary for São Paulo state
print("Creating São Paulo state boundary...")
sp_state_boundary = census_tracts.unary_union

# Ensure same CRS for data centers and state boundary
if data_centers.crs != census_tracts.crs:
    print(f"Converting data centers from {data_centers.crs} to {census_tracts.crs}")
    data_centers = data_centers.to_crs(census_tracts.crs)

# Filter data centers to those within the São Paulo state boundary
sp_data_centers = data_centers[data_centers.geometry.within(sp_state_boundary)]
print(f"Data centers in São Paulo state: {len(sp_data_centers)}")

# Get names if available
if 'name' in sp_data_centers.columns:
    print("\nData center names in São Paulo state:")
    for name in sp_data_centers['name']:
        print(f"- {name}")
elif 'title' in sp_data_centers.columns:
    print("\nData center names in São Paulo state:")
    for title in sp_data_centers['title']:
        print(f"- {title}")
else:
    print("\nNo name or title column found in data centers dataset.") 