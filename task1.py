# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx  # Added for network visualization
from classes import Network, Simulation

def create_example_network():
    """Create a sample genetic circuit."""
    network = Network()
    network.add_node("Gene", params={"promoter_strength": 1.0, "rbs_efficiency": 1.0})
    return network

def explore_parameter_space(network, gate_type):
    """
    Explore parameter space for the genetic circuit (OR or AND gate).
    :param network: Genetic circuit network.
    :param gate_type: Logic gate type ('OR' or 'AND').
    """
    simulation = Simulation(network, gate_type=gate_type)
    t = np.linspace(0, 50, 200)  # Simulate for 50 time units
    AHL_levels = np.linspace(0, 10, 10)  # Test AHL concentrations
    benzoate_levels = np.linspace(0, 10, 10)  # Test benzoate concentrations

    results = []
    for AHL in AHL_levels:
        for benzoate in benzoate_levels:
            params = {"AHL": AHL, "benzoate": benzoate, "K": 5.0, "n": 2.0}  # Add Hill parameters
            dynamics = simulation.simulate(params, t)
            final_state = dynamics[-1][0]  # Steady-state output
            results.append((AHL, benzoate, final_state))
    return results

def visualize_parameter_space(results, gate_type):
    """
    Visualize the parameter space for the circuit.
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

    sns.heatmap(output_matrix, xticklabels=np.round(benzoate_levels, 2),
                yticklabels=np.round(AHL_levels, 2), cmap="viridis",
                cbar_kws={"label": "Output Gene Expression"})
    plt.title(f"Parameter Space Heatmap ({gate_type} Gate)")
    plt.xlabel("Benzoate Level")
    plt.ylabel("AHL Level")
    plt.show()

def generate_networkx_graph(results):
    """
    Generate a networkx graph of the parameter space.
    Nodes represent specific parameter sets (AHL, benzoate), and edges connect similar states.
    """
    G = nx.Graph()

    # Add nodes with attributes
    for r in results:
        AHL, benzoate, output = r
        node_name = f"AHL={AHL:.2f}, Benzoate={benzoate:.2f}"
        G.add_node(node_name, AHL=AHL, benzoate=benzoate, output=output)

    # Add edges based on similarity in output levels
    nodes = list(G.nodes(data=True))
    for i, (node1, data1) in enumerate(nodes):
        for j, (node2, data2) in enumerate(nodes):
            if i < j:
                # Connect nodes with small differences in output
                if abs(data1['output'] - data2['output']) < 0.1:
                    G.add_edge(node1, node2)

    return G

def visualize_networkx_graph(G, gate_type):
    """
    Visualize the parameter space as a network graph.
    Node color represents output gene expression.
    """
    pos = nx.spring_layout(G)  # Use a spring layout for visualization
    outputs = nx.get_node_attributes(G, 'output')
    colors = [v for v in outputs.values()]  # Map outputs to colors

    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, 
                                   node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.colorbar(nodes, label="Gene Expression")  # Explicitly link colorbar to nodes
    plt.title(f"Parameter Space as Network Graph ({gate_type} Gate)")
    plt.axis("off")  # Turn off axis for clean visualization
    plt.show()

def plot_dynamic_response(network, gate_type, params):
    """
    Plot the dynamic response for specific AHL and benzoate levels.
    :param network: Genetic circuit network.
    :param gate_type: Logic gate type ('OR' or 'AND').
    :param params: Specific parameter set (e.g., AHL and benzoate levels).
    """
    simulation = Simulation(network, gate_type=gate_type)
    t = np.linspace(0, 50, 200)  # Simulate for 50 time units
    dynamics = simulation.simulate(params, t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, dynamics[:, 0], label="Output Gene Expression", color="blue")
    plt.title(f"Dynamic Response ({gate_type} Gate)\nAHL={params['AHL']}, Benzoate={params['benzoate']}")
    plt.xlabel("Time")
    plt.ylabel("Gene Expression")
    plt.grid()
    plt.legend()
    plt.show()

# %%
if __name__ == "__main__":
    network = create_example_network()
    
    # OR Gate
    print("Exploring OR Gate...")
    or_results = explore_parameter_space(network, gate_type="OR")
    visualize_parameter_space(or_results, "OR")
   # G_or = generate_networkx_graph(or_results)
   # visualize_networkx_graph(G_or, "OR")
    plot_dynamic_response(network, gate_type="OR", params={"AHL": 5.0, "benzoate": 2.0, "K": 5.0, "n": 2.0})

    # AND Gate
    print("Exploring AND Gate...")
    and_results = explore_parameter_space(network, gate_type="AND")
    visualize_parameter_space(and_results, "AND")
   # G_and = generate_networkx_graph(and_results)
   # visualize_networkx_graph(G_and, "AND")
    plot_dynamic_response(network, gate_type="AND", params={"AHL": 5.0, "benzoate": 5.0, "K": 5.0, "n": 2.0})
   

# %%
from graphviz import Digraph

def create_detailed_pipeline_graph():
    """
    Create a professional, structured graph representation of the ML-GA optimization pipeline.
    Includes both performed steps (solid lines) and intended future steps (dashed lines).
    """
    graph = Digraph(format="png", comment="ML-GA Optimization Workflow", graph_attr={'rankdir': 'TB'})

    # ----------------------------
    # Performed Steps
    # ----------------------------
    # Pre-Processing Stage
    graph.node("Sim_ODE", "Simulate ODE Dynamics\n(Time-Series Data)", shape="box", style="rounded,filled", fillcolor="lightgray")
    graph.node("Steady_State", "Extract Steady-State Outputs\n(For ML Features)", shape="box", style="rounded,filled", fillcolor="lightgray")
    graph.edge("Pre", "Sim_ODE", label="Completed", color="black", style="solid")
    graph.edge("Sim_ODE", "Steady_State", label="Completed", color="black", style="solid")

    # Machine Learning Stage
    graph.node("Train", "Train ML Model\n(Random Forest)", shape="box", style="rounded,filled", fillcolor="lightgray")
    graph.node("Predict", "Predict Circuit Behavior\n(ML Inference)", shape="box", style="rounded,filled", fillcolor="lightgray")
    graph.node("Init_Pop", "Initialize Population\n(Parameter Sets)", shape="box", style="rounded,filled", fillcolor="lightgray")
    graph.edge("Steady_State", "Train", label="Completed", color="black", style="solid")
    graph.edge("Train", "Predict", label="Completed", color="black", style="solid")

    # Genetic Algorithm Stage
    graph.node("GA", "Genetic Algorithm", shape="ellipse", style="filled", fillcolor="lightblue")
    graph.node("Fitness", "Evaluate Fitness\n(ML Model Predictions)", shape="box", style="rounded,filled", fillcolor="lightblue")
    graph.node("Crossover", "Crossover/Mutation\n(Generate New Parameters)", shape="box", style="rounded,filled", fillcolor="lightblue")
    graph.node("Selection", "Select Top Individuals", shape="box", style="rounded,filled", fillcolor="lightblue")
    graph.edge("Predict", "Fitness", label="Completed", color="black", style="solid")
    graph.edge("Fitness", "Crossover", label="Completed", color="black", style="solid")
    graph.edge("Crossover", "Selection", label="Completed", color="black", style="solid")
    graph.edge("Selection", "Fitness", label="Feedback Loop", color="black", style="solid")

    # ----------------------------
    # Intended Future Steps
    # ----------------------------
    # Promoter Evaluation Stage (Future Steps)
    graph.node("Fetch", "Fetch Promoter Library\n(SynBioHub)", shape="box", style="rounded,filled", fillcolor="lightyellow")
    graph.node("Match", "Match Promoters\n(Optimized Parameters)", shape="box", style="rounded,filled", fillcolor="lightyellow")
    graph.node("Simulate_Promoter", "Simulate Promoter Response\n(Input Signals)", shape="box", style="rounded,filled", fillcolor="lightyellow")
    graph.edge("Fetch", "Match", label="Planned", color="black", style="dashed")
    graph.edge("Match", "Simulate_Promoter", label="Planned", color="black", style="dashed")
    graph.edge("Selection", "Match", label="Completed", color="black", style="solid")

    # ODE Validation (Planned Step)
    graph.node("ODE_Validation", "ODE Validation\n(Validate Circuit Behavior)", shape="box", style="rounded,filled", fillcolor="lightyellow")
    graph.edge("Simulate_Promoter", "ODE_Validation", label="Planned", color="black", style="dashed")

    # Final Output
    graph.node("Output", "Final Optimized Circuit\n(AND/OR Gates)", shape="ellipse", style="filled", fillcolor="green")
    graph.edge("ODE_Validation", "Output", label="Planned", color="black", style="solid")

    return graph


# Render and visualize the graph
workflow_graph = create_detailed_pipeline_graph()
workflow_graph.render("Detailed_ML_GA_Optimization_Pipeline", view=False)
print("Graph saved as 'Detailed_ML_GA_Optimization_Pipeline.png'")

# %%
