#%%
import numpy as np
import os
import sbol3
import xml.etree.ElementTree as ET
import requests
import matplotlib.pyplot as plt
#from celloapi2 import CelloResult
from celloclass import CelloHandler
from task2 import train_ml_model, optimize_circuit, simulate_odes, plot_steady_state_dynamics, visualize_optimized_parameters
import seaborn as sns


def explore_parameter_space_with_cello(params, gate_type):
    """
    Explore parameter space for the genetic circuit (OR or AND gate) with Cello results.
    Biological Relevance:
    - Models gene expression dynamics across a range of AHL and Benzoate concentrations.
    - Highlights functional parameter spaces for the circuit.
    :param params: Optimized Cello parameters (e.g., promoter strengths, RBS efficiencies).
    :param gate_type: Logic gate type ('OR' or 'AND').
    :return: Results containing AHL, Benzoate levels, and gene expression outputs.
    """
    t = np.linspace(0, 50, 200)  # Time range for ODE simulation
    AHL_levels = np.linspace(0, 10, 20)  # Test AHL concentrations
    benzoate_levels = np.linspace(0, 10, 20)  # Test Benzoate concentrations

    results = []
    for AHL in AHL_levels:
        for benzoate in benzoate_levels:
            test_params = params.copy()
            test_params.update({"AHL": AHL, "benzoate": benzoate})
            dynamics = simulate_odes(test_params, t, gate_type)
            steady_state = dynamics[-1][0]  # Steady-state gene expression
            results.append((AHL, benzoate, steady_state))
    return results

def visualize_parameter_space_with_heatmap(results, gate_type):
    """
    Visualize the parameter space as a heatmap.
    Biological Relevance:
    - Highlights regions of AHL and Benzoate concentrations that yield desired circuit behaviors.
    :param results: Results of parameter space exploration.
    :param gate_type: Logic gate type ('OR' or 'AND').
    """
    AHL_levels = sorted(set(r[0] for r in results))
    benzoate_levels = sorted(set(r[1] for r in results))
    output_matrix = np.zeros((len(AHL_levels), len(benzoate_levels)))

    for r in results:
        i = AHL_levels.index(r[0])
        j = benzoate_levels.index(r[1])
        output_matrix[i, j] = r[2]

    plt.figure(figsize=(10, 6))
    sns.heatmap(output_matrix, xticklabels=np.round(benzoate_levels, 2),
                yticklabels=np.round(AHL_levels, 2), cmap="viridis",
                cbar_kws={"label": "Gene Expression Output"})
    plt.title(f"Parameter Space Heatmap ({gate_type} Gate)")
    plt.xlabel("Benzoate Concentration")
    plt.ylabel("AHL Concentration")
    plt.show()


def generate_verilog_file(gate_type, output_file):
    """
    Generate a Verilog file describing the circuit logic for Cello.

    Biological Relevance:
    - Defines the logical behavior of the genetic circuit (e.g., AND or OR gate).
    - Provides the high-level specification for Cello to design circuits using genetic parts.
    :param gate_type: 'OR' or 'AND'.
    :param output_file: Path to save the Verilog file.
    """
    logic = {
        "OR": """
module main (
    input AHL,
    input Benzoate,
    output GFP
);
    assign GFP = !(pTac || pTet);  // Updated with promoters from JSON
endmodule
""",
        "AND": """
module main (
    input AHL,
    input Benzoate,
    output GFP
);
    assign GFP = !(pTac && pTet);  // Updated with promoters from JSON
endmodule
"""
    }
  
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    with open(output_file, 'w') as f:
        f.write(logic[gate_type])
    print(f"Verilog file for {gate_type} gate written to {output_file}")


def run_cello_query(verilog_file, output_directory):
    """
    Run a Cello query to generate a genetic circuit.

    Biological Relevance:
    - Maps the logical specification in Verilog to actual biological parts using Cello's libraries.
    - Outputs the optimized circuit design with assigned genetic parts (e.g., promoters, RBS, terminators).
    :param verilog_file: Path to the Verilog file.
    :param output_directory: Directory to save results.
    :return: CelloResult object containing circuit details.
    """
    try:
        cello_handler = CelloHandler(
            verilog_file=verilog_file,
            output_directory=output_directory,
            input_names=["AHL", "Benzoate"],
            output_names=["GFP"]
        )
        results = cello_handler.generate_circuit()
        return results
    except Exception as e:
        print(f"Error running Cello query: {e}")
        raise
    
def fetch_sbol_collection(collection_url, output_file="collection.sbol"):
    """
    Fetch an SBOL collection from SynBioHub and parse it into an SBOL3 document.
    :param collection_url: URL to the SBOL collection.
    :param output_file: File path to save the fetched SBOL file.
    :return: Parsed SBOL3 document.
    """
    print(f"Downloading SBOL collection from {collection_url}...")
    os.system(f"wget -O {output_file} '{collection_url}/sbol'")
    if not os.path.exists(output_file):
        raise ValueError(f"Failed to download SBOL file from {collection_url}")

    print(f"Parsing SBOL file: {output_file}...")
    sbol3.set_namespace("http://example.org")
    doc = sbol3.Document()
    doc.read(output_file)

    return doc
    
def parse_parts_library_from_sbol(file_path):
    """
    Parse an SBOL file to extract promoter library information.
    
    :param file_path: Path to the SBOL XML file.
    :return: Dictionary mapping promoter names to their parameters.
    """
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Define namespaces
        ns = {
            'sbol': 'http://sbols.org/v2#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'dcterms': 'http://purl.org/dc/terms/',
            'ns0': 'http://cellocad.org/Terms/cello#',
            'so': 'http://identifiers.org/so/',
        }

        parts_library = {
            "promoters": {},
            "RBS": {},
            "CDS": {},
            "terminators": {}
        }

        # Iterate through ComponentDefinition elements
        for comp_def in root.findall("sbol:ComponentDefinition", ns):
            try:
                # Check roles to determine the part type
                roles = comp_def.findall("sbol:role", ns)
                role_urls = [role.attrib.get(f"{{{ns['rdf']}}}resource", "") for role in roles]

                # Extract common properties
                name = comp_def.find("sbol:displayId", ns).text
                title = comp_def.find("dcterms:title", ns).text if comp_def.find("dcterms:title", ns) else "Unknown"

                # Promoters (SO:0000804)
                if any("SO:0000804" in url for url in role_urls):
                    K = float(comp_def.find("ns0:K", ns).text)
                    n = float(comp_def.find("ns0:n", ns).text)
                    ymax = float(comp_def.find("ns0:ymax", ns).text)
                    ymin = float(comp_def.find("ns0:ymin", ns).text)
                    x_off_threshold = float(comp_def.find("ns0:x_off_threshold", ns).text)
                    x_on_threshold = float(comp_def.find("ns0:x_on_threshold", ns).text)

                    K_range = (K - (x_off_threshold / 2), K + (x_on_threshold / 2))
                    n_range = (n - 0.3, n + 0.3)

                    parts_library["promoters"][name] = {
                        "K_range": K_range,
                        "n_range": n_range,
                        "ymax": ymax,
                        "ymin": ymin,
                        "title": title
                    }

                # RBS (SO:0000139)
                elif any("SO:0000139" in url for url in role_urls):
                    strength = float(comp_def.find("ns0:strength", ns).text)
                    parts_library["RBS"][name] = {
                        "strength": strength,
                        "title": title
                    }

                # CDS (SO:0000316)
                elif any("SO:0000316" in url for url in role_urls):
                    protein_product = comp_def.find("ns0:protein_product", ns).text
                    parts_library["CDS"][name] = {
                        "protein_product": protein_product,
                        "title": title
                    }

                # Terminators (SO:0000141)
                elif any("SO:0000141" in url for url in role_urls):
                    efficiency = float(comp_def.find("ns0:efficiency", ns).text)
                    parts_library["terminators"][name] = {
                        "efficiency": efficiency,
                        "title": title
                    }

            except Exception as e:
                print(f"Error parsing component {name}: {e}")

        return parts_library

    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required field or prefix: {e}")
    
def parse_all_parts_from_folder(folder_path):
    """
    Parse all SBOL files in a folder and merge their parts into a single library.
    :param folder_path: Path to the folder containing SBOL files.
    :return: Dictionary containing all parts grouped by type.
    """
    combined_library = {
        "promoters": {},
        "RBS": {},
        "CDS": {},
        "terminators": {}
    }

    # Iterate over all XML files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml"):  # Process only XML files
            file_path = os.path.join(folder_path, file_name)
            try:
                print(f"Parsing file: {file_path}")
                parts_library = parse_parts_library_from_sbol(file_path)

                # Merge parts into the combined library
                for part_type, parts in parts_library.items():
                    combined_library[part_type].update(parts)
            except Exception as e:
                print(f"Error parsing file {file_name}: {e}")

    return combined_library


# Match promoters to ML results
def match_promoters_to_ml_results(optimized_params, parts_library):
    """
    Match ML-optimized parameters to promoters in the parts library.
    :param optimized_params: Dictionary of ML-optimized parameters for gates (e.g., OR, AND).
    :param parts_library: Dictionary of parts grouped by type.
    :return: Dictionary mapping gates to matching promoters.
    """
    if "promoters" not in parts_library:
        raise ValueError("Promoters not found in the parts library.")

    promoter_library = parts_library["promoters"]  # Focus on promoters
    matches = {}

    for gate, params in optimized_params.items():
        matches[gate] = [
            promoter for promoter, properties in promoter_library.items()
            if properties["K_range"][0] <= params["K"] <= properties["K_range"][1]
            and properties["n_range"][0] <= params["n"] <= properties["n_range"][1]
        ]

    return matches

def simulate_promoter_response(promoter, input_signal, promoter_library):
    """
    Simulate the response of a promoter to an input signal.
    :param promoter: Name of the promoter.
    :param input_signal: Normalized input signal (0-1 range).
    :param promoter_library: Dictionary of promoters with their parameters.
    :return: Simulated output signal strength.
    """
    if promoter not in promoter_library:
        raise ValueError(f"Promoter '{promoter}' not found in the library.")

    params = promoter_library[promoter]
    return params["ymin"] + (params["ymax"] - params["ymin"]) * input_signal

def plot_promoter_responses(matches, promoter_library):
    """
    Plot promoter responses for each gate.
    :param matches: Dictionary of matched promoters.
    :param promoter_library: Promoter library with parameters.
    """
    for gate, promoters in matches.items():
        responses = [
            simulate_promoter_response(promoter, input_signal=0.7, promoter_library=promoter_library)
            for promoter in promoters
        ]
        plt.bar(promoters, responses, label=f"{gate} Promoters")
    
    plt.title("Promoter Responses at Input Signal 0.7")
    plt.ylabel("Response Strength")
    plt.legend()
    plt.show()
    
def fetch_promoters_from_sparql(endpoint_url):
    """
    Fetch promoter information from a SPARQL endpoint.
    
    :param endpoint_url: URL of the SPARQL endpoint (e.g., SynBioHub collection endpoint).
    :return: List of promoters with their displayId, name, and description.
    """
    # Define the SPARQL query
    sparql_query = """
    PREFIX sbol: http://sbols.org/v2#Promoter
    
    SELECT ?promoter ?name ?description
    WHERE {
        ?promoter a sbol:Component ;
                  sbol:role <http://sbols.org/v2#Promoter> ;
                  sbol:displayId ?name ;
                  sbol:description ?description .
    }
    """
    
    headers = {"Accept": "application/json"}
    response = requests.post(
        endpoint_url,
        data={"query": sparql_query},
        headers=headers
    )
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch promoters: {response.status_code} {response.text}")
    
    # Parse the JSON response
    results = response.json()["results"]["bindings"]
    promoter_list = [
        {
            "promoter_iri": result["promoter"]["value"],
            "name": result["name"]["value"],
            "description": result.get("description", {}).get("value", "No description")
        }
        for result in results
    ]
    
    return promoter_list

def parse_parts_library_from_directory(directory):
    """
    Parse a single SBOL file from the given directory.
    Assumes the file to parse is the first XML file found.
    
    :param directory: Path to the directory containing SBOL XML files.
    :return: Dictionary of parts grouped by type.
    """
    try:
        # List all XML files in the directory
        xml_files = [f for f in os.listdir(directory) if f.endswith(".xml")]

        if not xml_files:
            raise FileNotFoundError(f"No XML files found in directory: {directory}")

        # Use the first XML file found (you can adapt this to select specific files)
        sbol_file_path = os.path.join(directory, xml_files[0])
        print(f"Processing SBOL file: {sbol_file_path}")

        # Parse the selected SBOL file
        parts_library = parse_parts_library_from_sbol(sbol_file_path)
        return parts_library

    except FileNotFoundError as e:
        raise ValueError(f"Error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error while parsing directory: {e}")

#%%
if __name__ == "__main__":
    # Step 1: Train ML Model
    print("Training ML model...")
    model = train_ml_model()
    
    # Initialize CelloHandler with example parameters (update paths as necessary)
    cello_handler = CelloHandler(
        verilog_file="resources/AND_gate.v",
        output_directory="output",
        input_names=["AHL", "Benzoate"],
        output_names=["GFP"]
    )
 
    #%% Step 2: Optimize OR Gate
    print("Optimizing OR Gate...")
    or_params, t_or, or_dynamics = optimize_circuit("OR", model)
    print("OR Gate Optimized Parameters:", or_params)

    #%% Step 3: Optimize AND Gate
    print("Optimizing AND Gate...")
    and_params, t_and, and_dynamics = optimize_circuit("AND", model)
    print("AND Gate Optimized Parameters:", and_params)
         
    #%% Step 4: Explore Parameter Space and Plot Heatmap for OR Gate
    print("Exploring Parameter Space for OR Gate...")
    or_results = explore_parameter_space_with_cello(or_params, "OR")
    visualize_parameter_space_with_heatmap(or_results, "OR")

    # Step 5: Explore Parameter Space and Plot Heatmap for AND Gate
    print("Exploring Parameter Space for AND Gate...")
    and_results = explore_parameter_space_with_cello(and_params, "AND")
    visualize_parameter_space_with_heatmap(and_results, "AND")

    # Step 6: Plot and Compare
    print("Plotting Steady-State Dynamics...")
    plot_steady_state_dynamics(t_or, or_dynamics, t_and, and_dynamics, or_params, and_params)
    visualize_optimized_parameters(or_params, and_params)
    
    #%% Step 7: Generate Verilog Files for Cello
    print("Generating Verilog files...")
    generate_verilog_file("OR", "resources/OR_gate.v")
    generate_verilog_file("AND", "resources/AND_gate.v")

    # Step 8: Run Cello for OR and AND Gates
    print("Running Cello for OR Gate...")
    try:
        cello_results_or = run_cello_query("resources/OR_gate.v", "output/OR_gate")
        print("Cello Results for OR Gate:", cello_results_or)
    except Exception as e:
        print(f"Failed to generate circuit for OR Gate: {e}")

    print("Running Cello for AND Gate...")
    try:
        cello_results_and = run_cello_query("resources/AND_gate.v", "output/AND_gate")
        print("Cello Results for AND Gate:", cello_results_and)
    except Exception as e:
        print(f"Failed to generate circuit for AND Gate: {e}")

   #%%     
       #%% Step 3: Simulate Promoter Responses Dynamically
    def simulate_promoter_responses(matches, parts_library, input_signal=0.7):
        """
        Simulate promoter responses dynamically based on matches and input signal.
        :param matches: Dictionary mapping gates to matching promoters.
        :param parts_library: Dictionary of parts grouped by type.
        :param input_signal: Normalized input signal (default = 0.7).
        """
        if "promoters" not in parts_library:
            raise ValueError("Promoters not found in the parts library.")

        promoter_library = parts_library["promoters"]  # Focus on promoters

        for gate, promoters in matches.items():
            for promoter in promoters:
                if promoter not in promoter_library:
                    print(f"Warning: Promoter {promoter} not found in promoter library.")
                    continue

                # Extract promoter properties
                promoter_properties = promoter_library[promoter]
                ymin = promoter_properties["ymin"]
                ymax = promoter_properties["ymax"]

                # Simulate response based on the input signal
                response = ymin + (ymax - ymin) * input_signal
                print(f"Gate: {gate}, Promoter: {promoter}, Response (Input {input_signal}): {response}")
                
        # Assuming matches and promoter_library are defined
        plot_promoter_responses(matches, promoter_library)

    # Define the directory containing the parts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parts_dir = os.path.join(current_dir, "parts")

    # Ensure the directory exists
    if not os.path.exists(parts_dir):
        print(f"Error: Parts directory '{parts_dir}' does not exist.")
        exit

    try:
        # Parse the parts library from the directory
        parts_library = parse_parts_library_from_directory(parts_dir)

        # Print the extracted parts library
        print("Parts Library successfully parsed:")
        for part_type, parts in parts_library.items():
            print(f"{part_type.capitalize()}:")
            for name, details in parts.items():
                print(f"  - {name}: {details}")
    except Exception as e:
        print(f"Error parsing the parts directory: {e}")
            
    try:
        optimized_params = {
            "OR_gate": or_params,
            "AND_gate": and_params,
            }

        # Assuming parts_library has been parsed
        matches = match_promoters_to_ml_results(optimized_params, parts_library)
        print("Matching Promoters:", matches)

    except Exception as e:
        print(f"Error matching promoters: {e}")

#%%
import matplotlib.pyplot as plt
import numpy as np

# Data from optimization (replace with actual dynamic values from your ML pipeline)
or_fitness = [
    0.4782, 0.4782, 1.0055, 1.0022, 1.0022, 1.0022, 1.0022, 1.0022, 0.9185, 0.9078,
    0.9028, 0.9028, 0.9028, 0.9028, 0.9028, 0.8959, 0.8975, 0.8872, 0.8872, 0.8833
]

and_fitness = [
    1.1045, 1.4991, 1.4624, 1.4622, 1.4329, 1.4619, 1.4569, 1.4321, 1.4117, 1.4083,
    1.3930, 1.3769, 1.3668, 1.3459, 1.3454, 1.3445, 1.3032, 1.3001, 1.3001, 1.2270
]

# Initial population parameters for OR and AND gates
or_population = [
    {'AHL': 8.93, 'benzoate': 2.38, 'K': 2.77, 'n': 2.38},
    {'AHL': 3.39, 'benzoate': 8.23, 'K': 4.88, 'n': 1.51},
    {'AHL': 8.77, 'benzoate': 1.85, 'K': 5.64, 'n': 2.73},
    # Add other individuals here...
]

and_population = [
    {'AHL': 3.10, 'benzoate': 3.04, 'K': 3.60, 'n': 2.72},
    {'AHL': 7.45, 'benzoate': 7.65, 'K': 2.94, 'n': 2.45},
    {'AHL': 4.15, 'benzoate': 5.34, 'K': 5.33, 'n': 1.74},
    # Add other individuals here...
]

# Plot Fitness Progression for OR and AND Gates
def plot_fitness_progression(or_fitness, and_fitness):
    generations = np.arange(1, len(or_fitness) + 1)
    plt.figure(figsize=(10, 6))
    
    # OR Gate
    plt.plot(generations, or_fitness, label="OR Gate", color="blue", marker="o")
    plt.text(len(or_fitness), or_fitness[-1], f"Final: {or_fitness[-1]:.4f}", color="blue")
    
    # AND Gate
    plt.plot(generations, and_fitness, label="AND Gate", color="red", marker="o")
    plt.text(len(and_fitness), and_fitness[-1], f"Final: {and_fitness[-1]:.4f}", color="red")
    
    # Labels and legend
    plt.title("Fitness Progression Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Parameter Distribution in Initial Population
def plot_initial_population(or_population, and_population):
    # Unpack parameters for OR Gate
    or_ahl = [ind["AHL"] for ind in or_population]
    or_benzoate = [ind["benzoate"] for ind in or_population]
    or_k = [ind["K"] for ind in or_population]
    or_n = [ind["n"] for ind in or_population]
    
    # Unpack parameters for AND Gate
    and_ahl = [ind["AHL"] for ind in and_population]
    and_benzoate = [ind["benzoate"] for ind in and_population]
    and_k = [ind["K"] for ind in and_population]
    and_n = [ind["n"] for ind in and_population]
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot for OR Gate
    plt.subplot(2, 1, 1)
    plt.scatter(or_k, or_n, c="blue", label="OR Gate")
    plt.xlabel("K")
    plt.ylabel("n")
    plt.title("Initial Parameter Distribution for OR Gate")
    plt.grid(True)
    plt.legend()
    
    # Scatter plot for AND Gate
    plt.subplot(2, 1, 2)
    plt.scatter(and_k, and_n, c="red", label="AND Gate")
    plt.xlabel("K")
    plt.ylabel("n")
    plt.title("Initial Parameter Distribution for AND Gate")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize the Genetic Algorithm Optimization Process
plot_fitness_progression(or_fitness, and_fitness)
plot_initial_population(or_population, and_population)

# %%
