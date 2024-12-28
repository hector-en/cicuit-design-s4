import networkx as nx
import os
#from celloapi2 import CelloQuery, CelloResult
from networkx import DiGraph
import matplotlib.pyplot as plt


class CelloHandler:
    def __init__(self, verilog_file, output_directory, input_names, output_names):
        """
        Initialize the CelloHandler to manage circuit design and parsing.

        Biological Relevance:
        - Enables the design of genetic circuits based on logical operations (e.g., AND, OR gates).
        - Translates high-level specifications (Verilog) into biological part assignments (promoters, RBS).
        :param verilog_file: Path to the Verilog file describing circuit logic.
        :param output_directory: Path to save the results from Cello.
        :param input_names: List of input names (e.g., [AHL, Benzoate]).
        :param output_names: List of output names (e.g., [GFP]).
        """
        self.verilog_file = verilog_file
        self.output_directory = output_directory
        self.input_names = input_names
        self.output_names = output_names
        # Promoter Parameters
        self.promoter_params = {}

    @staticmethod
    def find_matching_promoters(optimized_params, promoter_library):
        """
        Find matching promoters based on ML-optimized K and n values.
        Biological Relevance:
        - Matches computationally optimized parameters to real biological parts.
        :param optimized_params: Dictionary of optimized K and n values for each gate.
        :param promoter_library: Library of promoters with parameter ranges.
        :return: Dictionary of matched promoters for each gate.
        """
        matches = {}
        for gate, params in optimized_params.items():
            matches[gate] = [
                promoter for promoter, properties in promoter_library.items()
                if properties["K_range"][0] <= params["K"] <= properties["K_range"][1]
                and properties["n_range"][0] <= params["n"] <= properties["n_range"][1]
            ]
        return matches

    def simulate_promoter_response(self, promoter, input_signal):
        """
        Simulate the response of a promoter to an input signal.
        Biological Relevance:
        - Models transcriptional output based on promoter properties (ymax, ymin).
        :param promoter: Name of the promoter (e.g., 'pTac').
        :param input_signal: Normalized input signal (0-1 range).
        :return: Simulated output signal strength.
        """
        if promoter not in self.promoter_params:
            raise ValueError(f"Promoter '{promoter}' not found in parameters.")
        
        params = self.promoter_params[promoter]
        return params["ymin"] + (params["ymax"] - params["ymin"]) * input_signal
    
    def generate_circuit(self):
        """
        Generate a genetic circuit using Cello.

        Biological Relevance:
        - Automates the assignment of biological parts such as promoters and RBS sequences to implement circuit logic.
        - Provides a computational framework for selecting parts based on experimental datasets.
        :return: CelloResult object containing circuit details.
        """
        
        # Ensure the output directory is an absolute path
        absolute_output_directory = os.path.abspath(self.output_directory)
        # Ensure the directory exists
        os.makedirs(absolute_output_directory, exist_ok=True)
        
        try:
            query = CelloQuery(
                input_directory="resources",
                output_directory=absolute_output_directory,
                verilog_file=self.verilog_file,
                input_sensors="cello/files/v2/input/EcoEco1C1G1T1.input.json",
                output_device="cello/files/v2/output/EcoEco1C1G1T1.output.json",
                input_ucf="cello/files/v2/ucf/EcoEco1C1G1T1.UCF.json",
                compiler_options="resources/compiler_options.csv"
            )
            results = query.get_results()
            print("Cello query executed successfully.")
            return results
        except Exception as e:
            print(f"Error running Cello query: {e}")
            raise

    @staticmethod
    def parse_cello_results(results):
        """
        Parse Cello results to extract node and edge parameters.

        Biological Relevance:
        - Maps the circuit design into quantitative properties such as promoter strength and RBS efficiency.
        - Ensures biological feasibility by incorporating measured part properties from Cello’s library.
        :param results: CelloResult object containing circuit details.
        :return: Dictionary of parameters for nodes and edges.
        """
        try:
            circuit_params = {}
            for part in results.parts:
                if part.type == "promoter":
                    circuit_params[part.name] = {"promoter_strength": part.strength}
                elif part.type == "rbs":
                    circuit_params[part.name] = {"rbs_efficiency": part.efficiency}
            print("Successfully parsed Cello results.")
            return circuit_params
        except Exception as e:
            print(f"Error parsing Cello results: {e}")
            raise

    @staticmethod
    def build_network(params):
        """
        Build a networkx DiGraph from parsed Cello parameters.

        Biological Relevance:
        - Constructs a graphical representation of the genetic circuit, where:
          - Nodes represent genetic parts (e.g., promoters, RBS).
          - Edges represent regulatory interactions (e.g., activation, repression).
        :param params: Parsed parameters from Cello.
        :return: networkx DiGraph object.
        """
        try:
            G = DiGraph()
            for node, attributes in params.items():
                G.add_node(node, **attributes)
            print("Network successfully built from Cello parameters.")
            return G
        except Exception as e:
            print(f"Error building network: {e}")
            raise

    @staticmethod
    def visualize_optimized_circuit(network, params, title):
        """
        Visualize the optimized genetic circuit using networkx.

        Biological Relevance:
        - Provides a clear representation of the circuit structure and its components.
        - Highlights the relationships between nodes (genetic parts) and how they contribute to circuit behavior.
        :param network: networkx DiGraph object representing the circuit.
        :param params: Dictionary of optimized parameters.
        :param title: Title for the plot.
        """
        try:
            pos = nx.spring_layout(network)
            labels = {node: f"{node}\n{data}" for node, data in network.nodes(data=True)}

            plt.figure(figsize=(8, 6))
            nx.draw(
                network,
                pos,
                with_labels=True,
                labels=labels,
                node_size=3000,
                node_color="skyblue",
                edge_color="gray",
                font_size=10
            )
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Error visualizing the circuit: {e}")
            raise
