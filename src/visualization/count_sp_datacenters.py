import geopandas as gpd
from shapely.ops import unary_union

# Load data
print("Loading data...")
census_tracts = gpd.read_file('data/model_input/brasil/sao_paulo_census_tracts_full.geojson')
data_centers = gpd.read_file('data/processed/brasil/data_centers.geojson')

print(f"Total census tracts: {len(census_tracts)}")
print(f"Total data centers: {len(data_centers)}")

# Filter to São Paulo city (using municipal code 3550308)
print("Filtering to São Paulo city...")
sp_city_tracts = census_tracts[census_tracts['CD_SETOR'].astype(str).str[:7] == '3550308']
print(f"São Paulo city census tracts: {len(sp_city_tracts)}")

# Ensure same CRS
if data_centers.crs != sp_city_tracts.crs:
    print(f"Converting data centers from {data_centers.crs} to {sp_city_tracts.crs}")
    data_centers = data_centers.to_crs(sp_city_tracts.crs)

# Create city boundary
sp_boundary = unary_union(sp_city_tracts.geometry)

# Filter data centers to those within the boundary
sp_data_centers = data_centers[data_centers.geometry.within(sp_boundary)]
print(f"São Paulo city data centers: {len(sp_data_centers)}") 