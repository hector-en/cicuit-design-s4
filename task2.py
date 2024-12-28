# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
from classes import Network, Simulation, Optimizer  
#from celloapi2 import CelloQuery, CelloResult
import sbol3
import matplotlib.pyplot as pltfrom 
from graphviz import Digraph

# Step 1: Create Circuit Logic as Directed Graph
def create_circuit_graph(gate_type):
    """
    Create a directed graph representing the genetic circuit logic.
    Biological Relevance:
    - Nodes represent genes, regulatory elements, or molecules.
    - Edges represent regulatory interactions (e.g., repression or activation).
    :param gate_type: 'OR' or 'AND' gate.
    """
    G = nx.DiGraph()

    # Add nodes (genes) with example parameters
    G.add_node("Gene", promoter_strength=1.0, rbs_efficiency=1.0)

    # Add edges based on logic type
    if gate_type == "OR":
        # OR logic: Output is repressed if AHL OR benzoate is present
        G.add_edge("AHL", "Gene", weight=1.0)  # AHL represses output
        G.add_edge("Benzoate", "Gene", weight=1.0)  # Benzoate represses output
    elif gate_type == "AND":
        # AND logic: Output is repressed if AHL AND benzoate are both present
        G.add_edge("AHL", "Gene", weight=0.5)  # AHL contributes partially
        G.add_edge("Benzoate", "Gene", weight=0.5)  # Benzoate contributes partially
    return G

def plot_steady_state_dynamics(t_or, or_dynamics, t_and, and_dynamics, or_params, and_params):
    """
    Plot steady-state gene expression dynamics for OR and AND gates with annotations.
    :param t_or: Time array for OR gate simulation.
    :param or_dynamics: Simulated dynamics for OR gate.
    :param t_and: Time array for AND gate simulation.
    :param and_dynamics: Simulated dynamics for AND gate.
    :param or_params: Optimized parameters for the OR gate.
    :param and_params: Optimized parameters for the AND gate.
    """
    plt.figure(figsize=(10, 6))

    # Plot dynamics for OR Gate
    plt.plot(t_or, or_dynamics[:, 0], label="OR Gate", color="blue")
    final_or_value = or_dynamics[-1, 0]  # Final steady-state value
    plt.text(
        t_or[-1], final_or_value, f"{final_or_value:.2f}",
        fontsize=10, color="blue", verticalalignment="bottom", horizontalalignment="right"
    )

    # Plot dynamics for AND Gate
    plt.plot(t_and, and_dynamics[:, 0], label="AND Gate", color="red")
    final_and_value = and_dynamics[-1, 0]  # Final steady-state value
    plt.text(
        t_and[-1], final_and_value, f"{final_and_value:.2f}",
        fontsize=10, color="red", verticalalignment="bottom", horizontalalignment="right"
    )

    # Add explanatory text for K and n
    plt.gca().text(
        1.02, 0.5,
        f"Parameters:\nOR Gate:\n  K = {or_params['K']:.2f}, n = {or_params['n']:.2f}\n"
        f"AND Gate:\n  K = {and_params['K']:.2f}, n = {and_params['n']:.2f}",
        transform=plt.gca().transAxes, fontsize=10, verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5)
    )

    # Add labels, title, and legend
    plt.title("Time Evolution of Gene Expression (Optimized Parameters)")
    plt.xlabel("Time")
    plt.ylabel("Gene Expression")
    plt.legend()
    plt.grid(True)
    plt.show()
  
def visualize_parameter_space(results, gate_type):
    """Visualize the parameter space results."""
    AHL_levels = sorted(set(r[0] for r in results))
    benzoate_levels = sorted(set(r[1] for r in results))
    output_matrix = np.zeros((len(AHL_levels), len(benzoate_levels)))
    for r in results:
        i = AHL_levels.index(r[0])
        j = benzoate_levels.index(r[1])
        output_matrix[i, j] = r[2]
    sns.heatmap(output_matrix, xticklabels=np.round(benzoate_levels, 2),
                yticklabels=np.round(AHL_levels, 2), cmap="viridis",
                cbar_kws={"label": "Output Gene Expression"})
    plt.title(f"Parameter Space Heatmap ({gate_type} Gate)")
    plt.xlabel("Benzoate Level")
    plt.ylabel("AHL Level")
    plt.show() 
  
def visualize_optimized_parameters(or_params, and_params):
    """
    Compare optimized parameters for OR and AND gates using a bar chart with annotations.
    :param or_params: Optimized parameters for the OR gate.
    :param and_params: Optimized parameters for the AND gate.
    """
    params = ['AHL', 'benzoate', 'K', 'n']
    values_or = [or_params[key] for key in params]
    values_and = [and_params[key] for key in params]

    x = range(len(params))
    width = 0.35  # Bar width

    plt.figure(figsize=(10, 6))
    plt.bar(x, values_or, width, label="OR Gate", alpha=0.7, color="blue")
    plt.bar([p + width for p in x], values_and, width, label="AND Gate", alpha=0.7, color="red")

    # Annotate bar values
    for i, (or_val, and_val) in enumerate(zip(values_or, values_and)):
        plt.text(i, or_val + 0.1, f"{or_val:.2f}", ha='center', fontsize=10, color="blue")
        plt.text(i + width, and_val + 0.1, f"{and_val:.2f}", ha='center', fontsize=10, color="red")

    # Add labels, title, and legend
    plt.xticks([p + width / 2 for p in x], params)
    plt.title("Optimized Parameter Comparison (OR vs AND Gates)")
    plt.ylabel("Parameter Values")
    plt.legend()
    plt.grid(True)
    plt.show()
            
# Step 2: Simulate ODE Dynamics
def simulate_odes(params, t, gate_type):
    """
    Simulate ODEs for gene expression dynamics using Hill functions.
    Biological Relevance:
    - Models time-dependent gene expression as a function of regulatory inputs.
    - Delegates ODE dynamics to the Simulation class's simulate method.
    :param params: Parameter set including Hill coefficients, AHL, and Benzoate.
    :param t: Time array for the simulation.
    :param gate_type: Logic gate type ('OR' or 'AND').
    """
    # Create a Simulation instance with the specified gate type
    simulation = Simulation(None, gate_type)

    # Run the simulation with the given parameters and time array
    return simulation.simulate(params, t)

# Step 3: Fitness Function
def fitness_function(params, t, gate_type, model):
    """
    Calculate fitness for a given parameter set based on ODE simulation and ML predictions.
    Biological Relevance:
    - Evaluates how well a parameter set satisfies the desired circuit behavior.
    - Uses steady-state gene expression as a proxy for output quality.
    :param params: Parameter set for the genetic circuit.
    :param t: Time array for simulation.
    :param gate_type: Logic gate type ('OR' or 'AND').
    :param model: Machine Learning model for fitness evaluation.
    """
    # Validate params
    required_keys = ['AHL', 'benzoate', 'K', 'n']
    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing parameter: {key}")
    # Simulate ODEs   
    dynamics = simulate_odes(params, t, gate_type)
    steady_state = dynamics[-10:].mean()  # Average over last 10 time points (steady state)
    prediction = model.predict([[params['AHL'], params['benzoate'], steady_state]])
    return -prediction[0]  # Negative because GA maximizes fitness

# Step 4: GA Optimization
def optimize_circuit(gate_type, model): # Genetic Alghoritm to optimize promoter strength, RBS,)
    """
    Optimize genetic circuit parameters using a Genetic Algorithm (GA).
    Biological Relevance:
    - Identifies parameter sets that maximize stability and desired circuit output.
    - Combines machine learning predictions with genetic evolution.
    :param gate_type: Logic gate type ('OR' or 'AND').
    :param model: ML model trained on ODE data.
    """
    t = np.linspace(0, 50, 200)  # Time range for ODE simulation
    optimizer = Optimizer(create_circuit_graph(gate_type), Simulation(None, gate_type))

    best_params = optimizer.optimize(
        generations=20,  # Number of generations
        population_size=10,  # Number of individuals in each generation
        fitness_fn=lambda params: fitness_function(
            params, 
            t=np.linspace(0, 50, 200),  # Time range for simulation
            gate_type="OR",  # OR gate (adjust if using AND)
            model=model  # Machine Learning model for fitness evaluation
        )  # Fitness function for optimization
    )

    # Simulate dynamics with the best parameters
    optimized_dynamics = simulate_odes(best_params, t, gate_type)
    return best_params, t, optimized_dynamics


#def generate_circuit_with_cello(verilog_file, output_directory, input_names, output_names):
#    """
#    Generate a genetic circuit using Cello.
#    :param verilog_file: Path to the Verilog file describing the circuit logic.
#    :param output_directory: Directory to save Cello output files.
#    :param input_names: Names of the input signals (e.g., AHL, Benzoate).
#    :param output_names: Names of the outputs (e.g., GFP).
#    :return: CelloResult object containing circuit details.
#    """
#    query = CelloQuery(
#        input_directory="resources",
#        output_directory=output_directory,
#        verilog_file=verilog_file,
#        input_names=input_names,
#        output_names=output_names
#    )
#    query.run()
#    return query.get_results()

# Step 5: Train ML Model
def train_ml_model():
    """
    Train an ML model (e.g., RandomForest) on synthetic ODE data for fitness evaluation.
    Biological Relevance:
    - Provides a computationally efficient proxy for ODE simulations during optimization.
    """
    X, y = [], []  # Features and targets
    t = np.linspace(0, 50, 200)  # Time range for ODE simulation
    for AHL in np.linspace(0, 10, 10):
        for benzoate in np.linspace(0, 10, 10):
            params = {"AHL": AHL, "benzoate": benzoate, "K": 5.0, "n": 2.0}
            dynamics = simulate_odes(params, t, "OR")  # Example: OR gate
            steady_state = dynamics[-10:].mean()  # Use steady-state as feature
            X.append([AHL, benzoate, steady_state])
            y.append(1.0 - steady_state)  # Target: Maximize activation (1 - repression)

    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# candidate promoter library
candidates_promoter_library = {
    "pTac": {"ymax": 2.8, "ymin": 0.0034, "K_range": (3.0, 4.0), "n_range": (2.0, 2.5)},
    "pTet": {"ymax": 4.4, "ymin": 0.0013, "K_range": (2.0, 3.0), "n_range": (1.5, 2.2)},
}

# Simulate promoter response
def simulate_promoter_response(promoter, input_signal, promoter_library= candidates_promoter_library):
    """
    Simulate the response of a promoter to an input signal.
    """
    if promoter not in promoter_library:
        raise ValueError(f"Promoter '{promoter}' not found in the library.")
    
    # Extract promoter properties
    params = promoter_library[promoter]
    ymin = params.get("ymin", 0.0)  # Default ymin if missing
    ymax = params.get("ymax", 1.0)  # Default ymax if missing
    
    return ymin + (ymax - ymin) * input_signal

def plot_promoter_responses(matches, promoter_library):
    """
    Plot promoter responses for each gate.
    :param matches: Dictionary of matched promoters.
    :param promoter_library: Promoter library with parameters.
    """
    for gate, promoter in candidates_promoter_library.items():
        responses = [
            simulate_promoter_response(promoter, input_signal=0.7, promoter_library=candidates_promoter_library)
            for promoter in promoter
        ]
        plt.bar(promoter, responses, label=f"{gate} Promoters")
    
    plt.title("Promoter Responses at Input Signal 0.7")
    plt.ylabel("Response Strength")
    plt.legend()
    plt.show()

def create_sbol3_circuit(parts_library, output_file):
    """
    Create an sbol3 representation of a genetic circuit (OR and AND gates)
    with valid Interactions and Participations.
    """
    # Initialize sbol3 Document
    sbol3.Config.setOption("validate", True)
    sbol3.setHomespace("http://examples.org")
    doc = sbol3.Document()

    # Define Components
    pTac = sbol3.ComponentDefinition("pTac", sbol3.BIOPAX_DNA)
    pTac.roles = [sbol3.SO_PROMOTER]
    doc.addComponentDefinition(pTac)

    pTet = sbol3.ComponentDefinition("pTet", sbol3.BIOPAX_DNA)
    pTet.roles = [sbol3.SO_PROMOTER]
    doc.addComponentDefinition(pTet)

    GFP = sbol3.ComponentDefinition("GFP", sbol3.BIOPAX_DNA)
    GFP.roles = [sbol3.SO_CDS]
    doc.addComponentDefinition(GFP)

    RBS = sbol3.ComponentDefinition("RBS", sbol3.BIOPAX_DNA)
    RBS.roles = [sbol3.SO_RBS]
    doc.addComponentDefinition(RBS)

    Terminator = sbol3.ComponentDefinition("Terminator", sbol3.BIOPAX_DNA)
    Terminator.roles = [sbol3.SO_TERMINATOR]
    doc.addComponentDefinition(Terminator)

    # Create OR Gate ModuleDefinition
    or_gate = sbol3.ModuleDefinition("OR_Gate")
    doc.addModuleDefinition(or_gate)

    # Add Functional Components to the OR Gate
    pTac_fc = or_gate.addFunctionalComponent("pTac_fc", pTac, sbol3.SBOL_ACCESS_PUBLIC, sbol3.SBOL_DIRECTION_FORWARD)
    pTet_fc = or_gate.addFunctionalComponent("pTet_fc", pTet, sbol3.SBOL_ACCESS_PUBLIC, sbol3.SBOL_DIRECTION_FORWARD)
    GFP_fc = or_gate.addFunctionalComponent("GFP_fc", GFP, sbol3.SBOL_ACCESS_PUBLIC, sbol3.SBOL_DIRECTION_FORWARD)

    # Add Interaction for OR Logic
    or_interaction = or_gate.addInteraction("OR_Interaction", sbol3.SBO_LOGICAL_OR)

    # Add Participations to the OR Interaction
    or_interaction.addParticipation("pTac_participation", pTac_fc.identity, sbol3.SBO_STIMULATOR)
    or_interaction.addParticipation("pTet_participation", pTet_fc.identity, sbol3.SBO_STIMULATOR)
    or_interaction.addParticipation("GFP_participation", GFP_fc.identity, sbol3.SBO_PRODUCT)

    # Repeat for AND Gate
    and_gate = sbol3.ModuleDefinition("AND_Gate")
    doc.addModuleDefinition(and_gate)

    # Add Functional Components to the AND Gate
    pTac_fc_and = and_gate.addFunctionalComponent("pTac_fc_and", pTac, sbol3.SBOL_ACCESS_PUBLIC, sbol3.SBOL_DIRECTION_FORWARD)
    pTet_fc_and = and_gate.addFunctionalComponent("pTet_fc_and", pTet, sbol3.SBOL_ACCESS_PUBLIC, sbol3.SBOL_DIRECTION_FORWARD)
    GFP_fc_and = and_gate.addFunctionalComponent("GFP_fc_and", GFP, sbol3.SBOL_ACCESS_PUBLIC, sbol3.SBOL_DIRECTION_FORWARD)

    # Add Interaction for AND Logic
    and_interaction = and_gate.addInteraction("AND_Interaction", sbol3.SBO_LOGICAL_AND)

    # Add Participations to the AND Interaction
    and_interaction.addParticipation("pTac_participation_and", pTac_fc_and.identity, sbol3.SBO_STIMULATOR)
    and_interaction.addParticipation("pTet_participation_and", pTet_fc_and.identity, sbol3.SBO_STIMULATOR)
    and_interaction.addParticipation("GFP_participation_and", GFP_fc_and.identity, sbol3.SBO_PRODUCT)

    # Write to file
    doc.write(output_file)
    print(f"sbol3 circuit saved to {output_file}")



    """
    Visualize the genetic circuit as a NetworkX graph.
    :param G: NetworkX graph.
    :param title: Title of the visualization.
    """
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, "name")

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos, with_labels=True, labels=labels, node_size=3000, node_color="lightblue"
    )
    plt.title(title)
    plt.show()


# %%
if __name__ == "__main__":
    print("Training ML model...")
    model = train_ml_model()

    # Optimize OR Gate
    print("Optimizing OR Gate...")
    or_params, t_or, or_dynamics = optimize_circuit("OR", model)

    # Optimize AND Gate
    print("Optimizing AND Gate...")
    and_params, t_and, and_dynamics = optimize_circuit("AND", model)
    
    #print("Exploring OR Gate...")
    #or_results = explore_parameter_space(network, gate_type="OR")
    #visualize_parameter_space(or_results, "OR")

    # Plot steady-state dynamics
    print("Plotting Steady-State Dynamics...")
    plot_steady_state_dynamics(t_or, or_dynamics, t_and, and_dynamics, or_params, and_params)

    # Plot optimized parameter comparison
    print("Plotting Optimized Parameter Comparison...")
    visualize_optimized_parameters(or_params, and_params)
    
  


import sbol3

# Step 1: Initialize an SBOL Document
sbol3.set_namespace("http://example.org")
doc = sbol3.Document()

# Step 2: Create Components
# Promoter
promoter = sbol3.Component(
    "pTac",  # Unique ID
    sbol3.SBO_DNA,  # Component type
)
promoter.name = "pTac Promoter"
promoter.roles.append("http://identifiers.org/so/SO:0000167")  # SO term for promoter
doc.add(promoter)

# RBS
rbs = sbol3.Component(
    "RBS",  # Unique ID
    sbol3.SBO_DNA,
)
rbs.name = "Ribosome Binding Site"
rbs.roles.append("http://identifiers.org/so/SO:0000139")  # SO term for RBS
doc.add(rbs)

# CDS
cds = sbol3.Component(
    "GFP",  # Unique ID
    sbol3.SBO_DNA,
)
cds.name = "Green Fluorescent Protein"
cds.roles.append("http://identifiers.org/so/SO:0000316")  # SO term for CDS
doc.add(cds)

# Terminator
terminator = sbol3.Component(
    "Terminator",
    sbol3.SBO_DNA,
)
terminator.name = "Terminator"
terminator.roles.append("http://identifiers.org/so/SO:0000141")  # SO term for terminator
doc.add(terminator)

# Step 3: Create a Composite Genetic Circuit
# Create the circuit (parent component)
circuit = sbol3.Component("Simple_Circuit", sbol3.SBO_DNA)
circuit.name = "Simple Genetic Circuit"
circuit.roles.append("http://identifiers.org/so/SO:0000804")  # SO term for engineered region
doc.add(circuit)

# Add SubComponents to represent the hierarchy
circuit.promoter = sbol3.SubComponent(promoter)
circuit.rbs = sbol3.SubComponent(rbs)
circuit.cds = sbol3.SubComponent(cds)
circuit.terminator = sbol3.SubComponent(terminator)

# Add subcomponents to the circuit
circuit.features.extend([circuit.promoter, circuit.rbs, circuit.cds, circuit.terminator])

# Step 4: Save the SBOL Document
output_file = "simple_circuit.rdf"  # Use .rdf extension for SBOL3 files
doc.write(output_file)
print(f"SBOL document written to {output_file}")

