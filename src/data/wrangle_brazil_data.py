import os
import pandas as pd
import geopandas as gpd
import logging
import time
import re
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm import tqdm
import unicodedata
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class GeocodingCache:
    """Simple cache for geocoding results"""
    def __init__(self):
        self.cache = {}
    
    def get(self, address):
        return self.cache.get(address)
    
    def set(self, address, coords):
        self.cache[address] = coords

def format_brazilian_address(address: str) -> str:
    """
    Format a Brazilian address for better geocoding results.
    
    Args:
        address: The address to format
    
    Returns:
        A formatted address string
    """
    if not address:
        return address
        
    # Remove any text within parentheses, which can sometimes confuse the geocoder
    address = re.sub(r'\(.*?\)', '', address)
    
    # Replace known separators with commas, and remove redundant parts
    address = address.replace(' - State of ', ', ')
    address = address.replace('State of ', '')
    address = address.replace(' - ', ', ')
    
    # Normalize state abbreviations
    address = re.sub(r', (SP|RJ|RS|PR|CE|BA|DF|ES|PB|PE|SC|GO|AM|PA|RN|MT|MS|AL|MA|PI|SE|RO|TO|AC|RR|AP), ', ', ', address)
    
    # Standardize common terms
    address = address.replace('Avenida', 'Av.')
    address = address.replace('Rua', 'R.')
    address = address.replace('Estrada', 'Estr.')
    address = address.replace('Rodovia', 'Rod.')
    
    # Remove extra spaces and commas
    address = re.sub(r'\s+', ' ', address)
    address = re.sub(r',\s*,', ',', address)
    address = address.strip()
    
    # Ensure Brazil is at the end
    if not address.lower().endswith('brazil') and not address.lower().endswith('brasil'):
        address = f"{address}, Brazil"
    
    return address

def normalize_address(address):
    """
    Normalize an address for better matching:
    - Convert to lowercase
    - Remove accents
    - Replace common abbreviations
    - Remove zip codes
    - Remove extra spaces
    """
    if not address:
        return ""
    
    # Convert to lowercase
    address = address.lower()
    
    # Remove accents
    address = unicodedata.normalize('NFKD', address).encode('ASCII', 'ignore').decode('utf-8')
    
    # Replace common abbreviations
    replacements = {
        'av.': 'avenida',
        'av ': 'avenida ',
        'r.': 'rua',
        'r ': 'rua ',
        'rod.': 'rodovia',
        'estr.': 'estrada',
        'st.': 'state',
        'st ': 'state ',
        'res.': 'residencial',
        'res ': 'residencial ',
        'tambore': 'tambore',  # Keep tambore consistent
        'state of sao paulo': 'sao paulo',
        'sp': 'sao paulo',
        'rj': 'rio de janeiro',
        'ce': 'ceara',
        'rs': 'rio grande do sul'
    }
    
    for old, new in replacements.items():
        address = address.replace(old, new)
    
    # Remove zip codes
    address = re.sub(r'\d{5}-\d{3}', '', address)
    address = re.sub(r'\d{8}', '', address)
    
    # Remove extra spaces
    address = re.sub(r'\s+', ' ', address).strip()
    
    return address

def find_best_match_for_address(input_address, coordinates_dict, threshold=80):
    """
    Find the best match for an address in the coordinates dictionary.
    Returns (coordinates, matched_key) if a good match is found, otherwise (None, None).
    
    Args:
        input_address: The address to match
        coordinates_dict: Dictionary of address -> coordinates
        threshold: Minimum score (0-100) required for a match
        
    Returns:
        Tuple of (coordinates, matched_key) or (None, None) if no match
    """
    if not input_address:
        return None, None
    
    normalized_input = normalize_address(input_address)
    
    best_score = 0
    best_match = None
    best_key = None
    
    for key, coords in coordinates_dict.items():
        normalized_key = normalize_address(key)
        
        # Skip if key is too different in length (optimization)
        if abs(len(normalized_key) - len(normalized_input)) > 20:
            continue
            
        # Calculate similarity score
        similarity = SequenceMatcher(None, normalized_input, normalized_key).ratio() * 100
        
        # Check if this is our best match so far
        if similarity > best_score:
            best_score = similarity
            best_match = coords
            best_key = key
    
    # Return the best match if it exceeds our threshold
    if best_score >= threshold:
        return best_match, best_key
    
    return None, None

# Comprehensive hardcoded coordinates dictionary
addresses_coordinates = {
    "Avenida Marcos Penteado de Ulhôa Rodrigues, 249 - Residencial Tres (Tambore), Santana de Parnaíba - State of São Paulo, Brazil": (-23.46606308068713, -46.863222974485105),
    "Avenida Eid Mansur, 666 - Vila Santo Antonio de Carapicuiba, Cotia - State of São Paulo, Brazil": (-23.597718777057807, -46.84882763215363),
    "Rua do Semeador, 350 - Cidade Industrial de Curitiba, Curitiba - State of Paraná, Brazil": (-25.469954972071427, -49.350227745597444),
    "Rua Guido de Camargo Penteado Sobrinho - Chácara de Recreio Barao, Campinas - State of São Paulo, Brazil": (-22.822487290611765, -47.09737861682907),
    "Rua Bento Branco de Andrade Filho - Jardim Dom Bosco, São Paulo - State of São Paulo, Brazil": (-23.650327826125505, -46.72091275280389),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 1690 - Res. Tres (Tambore), Santana de Parnaíba - State of São Paulo, Brazil": (-23.46805076519034, -46.85837508797739),
    "Av. Eid Mansur, 666 - Vila Santo Antonio de Carapicuiba, Cotia - São Paulo, Brazil": (-23.597718777057807, -46.84882763215363),
    "Avenida Marcos Penteado de Ulhôa Rodrigues - Residencial Tres (Tambore), Santana de Parnaíba - State of São Paulo, Brazil": (-23.46606308068713, -46.863222974485105),
    "R. Dr. Miguel Couto, 58 - Centro Histórico de São Paulo, São Paulo - SP, 01008-010, Brazil": (-23.54579396102029, -46.63581750331895),
    "Av. Roberto Pinto Sobrinho, 306-744 - Vila Menck, Osasco - SP, 02675-031, Brazil": (-23.49120257499191, -46.77620671710327),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 888 - Residencial Tres (Tambore), Santana de Parnaíba - São Paulo, 06543-001, Brazil": (-23.46354585334552, -46.86263861681333),
    "Avenida Marginal Direita Anchieta, 1241 - Jordanópolis, São Bernardo do Campo - State of São Paulo, Brazil": (-23.67661126464669, -46.57224368056414),
    "Avenida Marcos Penteado de Ulhôa Rodrigues, 249 - Residencial Tres (Tambore), Santana de Parnaíba - State of São Paulo, 06543-385, Brazil": (-23.46606308068713, -46.863222974485105),
    "Avenida Pierre Simon de Laplace, 1211 - Techno Park, Campinas - SP, 13069-320, Brazil": (-22.847857964541213, -47.15341293032094),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 249 - Res. Tres (Tambore), Santana de Parnaíba - SP, Brazil": (-23.46606308068713, -46.863222974485105),
    "Av. Eid Mansur, 666 - Vila Santo Antonio de Carapicuiba, Cotia - State of São Paulo, Brazil": (-23.597718777057807, -46.84882763215363),
    "Rod. Pres. Dutra, 4648 - Jardim Jose Bonifacio, São João de Meriti - RJ, 25565-350, Brasil": (-22.79765685269606, -43.35712164434998),
    "Alameda Glete, 700 - Campos Elíseos, São Paulo - State of São Paulo, Brazil": (-23.535631134030776, -46.647555674483314),
    "Av. Roberto Pinto Sobrinho, 350 - Vila Menk, Osasco - São Paulo, Brazil": (-23.492062750026932, -46.77559996099191),
    "Av. Cid Viêira da Souza, 460-484 - Colinas da Anhanguera, Santana de Parnaíba - State of São Paulo, 06544, Brazil": (-23.461057413139304, -46.86039510093452),
    "Rua Voluntários da Pátria, 1555 - Historical Centre, Porto Alegre - RS, Brazil": (-30.02014547349709, -51.21412577430132),
    "Av. Beirute - Jardim Ermida I, Jundiaí - State of São Paulo, Brazil": (-23.193242960597118, -46.97303158009073),
    "Ascenty - São Paulo - Avenida Roberto Pinto Sobrinho, 350 - Vila Osasco, São Paulo - SP, 06268-120, Brazil": (-23.491872985024322, -46.778284760992065),
    "Av. Pierre Simon de Laplace, 1211 - Techno Park, Campinas - SP, 13069-320, Brazil": (-22.847857964541213, -47.15341293032094),
    "Rua Raimundo Esteves, 333 - Antonio Diogo, Fortaleza - Ceará, 60182-330, Brazil": (-3.735141541385463, -38.462394068638574),
    "Ascenty Sumaré - Rod. Anhanguera, s/n - Parque das Industrias (Nova Veneza), Sumaré - State of São Paulo, Brazil": (-22.812887008667722, -47.21344423032169),
    "R. José Blumer, 150 - Chácaras Assay, Hortolândia - SP, 13186-510, Brazil": (-22.899332213490894, -47.195271832170704),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 1690 - Res. Tambore III, Santana de Parnaíba - SP, 06543-001, Brazil": (-23.46805076519034, -46.85837508797739),
    "Avenida Marginal Direita Anchieta, 1241 - Jordanópolis, São Bernardo do Campo - State of São Paulo, Brazil": (-23.677161867425834, -46.56930393030046),
    "Quinta da Boa Vista - Parque Quinta da Boa Vista - São Cristóvão, Rio de Janeiro - RJ, Brazil": (-22.90460120295819, -43.217891135621244),
    "Parque das Indústrias, Paulínia - State of São Paulo, Brazil": (-22.771036551628416, -47.10651880027636),
    "R. Maria Soares Sendas - Parque Barreto, São João de Meriti - State of Rio de Janeiro, 25586-140, Brazil": (-22.797618512895472, -43.356554874459064),
    "Condomínio Panamérica Park - Av. Guido Caloi - Jardim São Luís, São Paulo - State of São Paulo, Brazil": (-23.653224371462926, -46.726076557286035),
    "Av. Roberto Pinto Sobrinho, 306-744 - Vila Menck, Osasco - SP, 02675-031, Brazil": (-23.489972613783372, -46.77476905309836),
    "Av. Ceci, 1900 - Res. Tambore, Barueri - SP, 06460-120, Brazil": (-23.49257928347365, -46.808420916812665),
    "Setor Comercial Sul Quadra 6 Renovação CNH Brasília DF - Clínica Credenciada Detran DF - Dom Bosco, Venâncio Shopping - Asa Sul, Brasília - Distrito Federal, Brazil": (-15.788014056279364, -47.88480957812323),
    "Avenida João Batista Falssarela - Jardim dos Passaros, Vinhedo - State of São Paulo, Brazil": (-23.070231210791935, -47.01165492420483),
    "Rua Bento Branco de Andrade Filho, 621 - Jardim Dom Bosco, São Paulo - SP, 04757-000, Brasil": (-23.650384907532015, -46.72099961680875),
    "G54R+H7 Barueri, State of São Paulo, Brazil": (-23.494910479016497, -46.811190049595474),
    "Avenida João Batista Falssarela - Jardim dos Passaros, Vinhedo - State of São Paulo, Brazil": (-23.070231210791935, -47.01165492420483),
    "Avenida Marcos Penteado de Ulhôa Rodrigues, 249 - Residencial Tres (Tambore), Santana de Parnaíba - State of São Paulo, 06543-001, Brazil": (-23.46606308068713, -46.863222974485105),
    "R. Vargem Grande, 100 - Jacarepaguá, Rio de Janeiro - State of Rio de Janeiro, Brazil": (-22.967167305567237, -43.380176080197046),
    "Estr. dos Romeiros, 39 - Votuparim, Santana de Parnaíba - SP, 06513-001, Brazil": (-23.45483887424143, -46.9150742879776),
    "Rod. Pres. Dutra - Venda Velha, São João de Meriti - RJ, 25565-350, Brazil": (-22.797621583028015, -43.35710757532943),
    "Distrito Industrial Benedito Storani, Vinhedo - State of São Paulo, Brazil": (-23.070754792054746, -47.011382405446064),
    "Alameda Glete, 700 - Campos Elíseos, São Paulo - SP, 01215-001, Brazil": (-23.535631134030776, -46.647555674483314),
    "Distrito Industrial Benedito Storani, Vinhedo - State of São Paulo, Brazil": (-23.070754792054746, -47.011382405446064),
    "Via Anhanguera, 2480 - Parque das Industrias (Nova Veneza), Sumaré - SP, Brazil": (-22.81294701011119, -47.21348477050851),
    "SCN Q 3 - Asa Norte, Brasilia - Federal District, Brazil": (-15.788188781464036, -47.884892673865735),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 1690 - Residencial Tambore III, Santana de Parnaíba - São Paulo, 06543-001, Brazil": (-23.46805076519034, -46.85837508797739),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 1690 - Residencial Tambore III, Santana de Parnaíba - São Paulo, 06543-001, Brazil": (-23.46805076519034, -46.85837508797739),
    "Av. Marcos Penteado de Ulhôa Rodrigues, 1690 - Residencial Tambore III, Santana de Parnaíba - São Paulo, 06543-001, Brazil": (-23.46805076519034, -46.85837508797739),
    "Rua Raimundo Esteves, 333 - Antonio Diogo, Fortaleza - CE, Brazil": (-3.734702593381458, -38.458070347787995),
    "Rua Santa Teresa, 64 - Centro Histórico de São Paulo, São Paulo - State of São Paulo, Brazil": (-23.54576357692257, -46.63587280474108),
    "R. Raimundo Esteves, 333 - Antonio Diogo, Fortaleza - CE, 60182-330, Brazil": (-3.734702593381458, -38.458070347787995),
    "Rodovia Jornalista Francisco Aguirre Proença, Monte Mor - State of São Paulo, Brazil": (-22.899396475637403, -47.195314744177665)
}

def geocode_address(address, max_retries=3):
    """
    Geocode an address with retries, caching, and validation.
    
    Args:
        address: The address to geocode
        max_retries: Maximum number of retry attempts
    
    Returns:
        A tuple of (latitude, longitude) or None if geocoding failed
    """
    # Check if the address has hardcoded coordinates
    if address in addresses_coordinates:
        coords = addresses_coordinates[address]
        logging.info(f"Using hardcoded coordinates for address: {address} -> {coords}")
        return coords
    
    # Create a geocoder with appropriate user agent
    geocoder = Nominatim(user_agent="brazil_dc_prediction_contact@example.com")
    
    # Initialize cache if not already created
    if not hasattr(geocode_address, 'cache'):
        geocode_address.cache = GeocodingCache()
    
    cache = geocode_address.cache
    
    # Check if we have already geocoded this address
    cached_coords = cache.get(address)
    if cached_coords:
        logging.info(f"Cache hit for address: {address}")
        return cached_coords
    
    # Format the address for better results
    formatted_address = format_brazilian_address(address)
    
    # Brazil's bounding box (rough approximation)
    # Latitude ranges from about -33.75 (south) to 5.25 (north)
    # Longitude ranges from about -73.99 (west) to -34.80 (east)
    BRAZIL_BOUNDS = {
        'min_lat': -33.75,
        'max_lat': 5.25,
        'min_lon': -73.99,
        'max_lon': -34.80
    }
    
    for attempt in range(max_retries):
        try:
            # First try with the formatted address including country and locale details
            location = geocoder.geocode(
                formatted_address, 
                timeout=10, 
                country_codes="br", 
                addressdetails=True
            )
            
            if not location and formatted_address != address:
                # If that fails, try the original address as a fallback
                location = geocoder.geocode(
                    address, 
                    timeout=10, 
                    country_codes="br", 
                    addressdetails=True
                )
            
            if location:
                coords = (location.latitude, location.longitude)
                # Validate that coordinates fall within Brazil's boundaries
                if (BRAZIL_BOUNDS['min_lat'] <= coords[0] <= BRAZIL_BOUNDS['max_lat'] and 
                    BRAZIL_BOUNDS['min_lon'] <= coords[1] <= BRAZIL_BOUNDS['max_lon']):
                    cache.set(address, coords)
                    logging.info(f"Successfully geocoded address: {address} -> {coords}")
                    return coords
                else:
                    logging.warning(f"Coordinates appear outside of Brazil bounds: {address} -> {coords}")
            else:
                logging.warning(f"No location found for address: {address}")
                
            # Respect Nominatim's usage policy with backoff
            time.sleep(1 + attempt)  # Increasing delay with each attempt
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logging.error(f"Attempt {attempt + 1} failed for address '{address}': {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All attempts failed for address: {address}")
    
    return None

def process_data_centers(src_path, dest_path, test_mode=False):
    """
    Process data center locations.
    
    Args:
        src_path: Path to the source CSV file
        dest_path: Path to save processed data
        test_mode: If True, only process a few rows
    """
    logging.info(f"Loading data center locations from: {src_path}")
    data = pd.read_csv(src_path)
    
    # If test mode, only use first 10 rows
    if test_mode:
        data = data.head(10)
    
    # Add columns for coordinates
    data['latitude'] = None
    data['longitude'] = None
    
    # Geocode addresses
    logging.info("Starting geocoding process...")
    with tqdm(total=len(data), desc="Geocoding addresses") as pbar:
        for idx, row in data.iterrows():
            address = row['address']
            coords = geocode_address(address)
            
            if coords:
                data.at[idx, 'latitude'] = coords[0]
                data.at[idx, 'longitude'] = coords[1]
            
            pbar.update(1)
    
    # Filter to only include rows with coordinates
    geocoded_data = data.dropna(subset=['latitude', 'longitude'])
    
    # Log the results
    failed_addresses = data[data['latitude'].isna()]['address'].tolist()
    logging.info(f"Geocoding completed. Success: {len(geocoded_data)}, Failed: {len(failed_addresses)}")
    
    if failed_addresses:
        logging.warning("Failed addresses:")
        for addr in failed_addresses:
            logging.warning(f"- {addr}")
    
    # Convert to GeoDataFrame
    if len(geocoded_data) > 0:
        gdf = gpd.GeoDataFrame(
            geocoded_data, 
            geometry=gpd.points_from_xy(geocoded_data.longitude, geocoded_data.latitude),
            crs="EPSG:4326"
        )
        
        # Save to file
        os.makedirs(dest_path, exist_ok=True)
        gdf.to_file(os.path.join(dest_path, "data_centers.geojson"), driver="GeoJSON")
        
        logging.info(f"Created {len(gdf)} records")
        logging.info(f"Created {len(gdf)} records")
        logging.info(f"Saved processed data centers to: {dest_path}")
        
        return gdf
    else:
        logging.warning("No data centers could be geocoded.")
        return None

def process_cell_towers(src_path, dest_path, test_mode=False):
    """
    Process cell tower locations.
    
    Args:
        src_path: Path to the source CSV file
        dest_path: Path to save processed data
        test_mode: If True, only process a small subset
    """
    logging.info(f"Loading cell towers data from: {src_path}")
    data = pd.read_csv(src_path)
    
    # Filter to Brazil only using MCC 724 (Brazil's Mobile Country Code)
    brazil_data = data[data['mcc'] == 724]
    logging.info(f"Filtered to {len(brazil_data)} cell towers in Brazil")
    
    # If test mode, only use a sample
    if test_mode:
        brazil_data = brazil_data.sample(min(100, len(brazil_data)))
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        brazil_data, 
        geometry=gpd.points_from_xy(brazil_data.lon, brazil_data.lat),
        crs="EPSG:4326"
    )
    
    # Save to file
    os.makedirs(dest_path, exist_ok=True)
    gdf.to_file(os.path.join(dest_path, "cell_towers.geojson"), driver="GeoJSON")
    
    logging.info(f"Created {len(gdf):,} records")
    logging.info(f"Saved processed cell towers to: {dest_path}")
    
    return gdf

def process_infrastructure(src_substations, src_transmission_lines, dest_path, test_mode=False):
    """
    Process electrical infrastructure data (substations and transmission lines).
    
    Args:
        src_substations: Path to substation GeoJSON file
        src_transmission_lines: Path to transmission lines GeoJSON file
        dest_path: Path to save processed data
        test_mode: If True, only process a small subset
    """
    logging.info(f"Loading substations from: {src_substations}")
    substations = gpd.read_file(src_substations)
    
    logging.info(f"Loading transmission lines from: {src_transmission_lines}")
    transmission_lines = gpd.read_file(src_transmission_lines)
    
    # If test mode, only use a sample
    if test_mode:
        substations = substations.sample(min(20, len(substations)))
        transmission_lines = transmission_lines.sample(min(20, len(transmission_lines)))
    
    # Save to files
    os.makedirs(dest_path, exist_ok=True)
    substations.to_file(os.path.join(dest_path, "substations.geojson"), driver="GeoJSON")
    transmission_lines.to_file(os.path.join(dest_path, "transmission_lines.geojson"), driver="GeoJSON")
    
    logging.info(f"Created {len(substations)} records")
    logging.info(f"Created {len(transmission_lines)} records")
    logging.info(f"Saved processed infrastructure data to: {dest_path}")
    
    return substations, transmission_lines

def main(test_mode=False):
    """
    Main function to process all datasets.
    
    Args:
        test_mode: If True, only process a small subset of data
    """
    logging.info(f"Starting data processing for Brazil{' in test mode' if test_mode else ''}...")
    
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    
    # Make sure we're using the correct Google Drive path
    base_dir_str = str(base_dir)
    if "GoogleDrive-leonardogomes@gmail.com" in base_dir_str:
        correct_drive = "GoogleDrive-leonardogomes@gmail.com"
    else:
        correct_drive = "GoogleDrive-leodavinci550@gmail.com"
    
    # Ensure we use consistent paths
    logging.info(f"Using Google Drive path: {correct_drive}")
    
    # Data center locations
    dc_src = os.path.join(base_dir, "data/raw/brasil/datacenter_map_scraped_brasil.csv")
    
    # Cell towers
    cell_towers_src = os.path.join(base_dir, "data/raw/global/cell_towers.csv")
    
    # Electrical infrastructure
    substations_src = os.path.join(base_dir, "data/raw/brasil/infraestrutura_eletrica/subestacao.geojson")
    transmission_lines_src = os.path.join(base_dir, "data/raw/brasil/infraestrutura_eletrica/linha_transmissao.geojson")
    
    # Destination for processed data
    dest_path = os.path.join(base_dir, "data/processed/brazil")
    
    # Log all paths for verification
    logging.info(f"Data centers source: {dc_src}")
    logging.info(f"Cell towers source: {cell_towers_src}")
    logging.info(f"Substations source: {substations_src}")
    logging.info(f"Transmission lines source: {transmission_lines_src}")
    logging.info(f"Destination path: {dest_path}")
    
    # Process each dataset
    process_data_centers(dc_src, dest_path, test_mode)
    process_cell_towers(cell_towers_src, dest_path, test_mode)
    process_infrastructure(substations_src, transmission_lines_src, dest_path, test_mode)
    
    logging.info("Data processing completed successfully!")

if __name__ == "__main__":
    main(test_mode=False) 