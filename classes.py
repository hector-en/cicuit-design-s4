import networkx as nx
import numpy as np
from scipy.integrate import odeint
import random

# Define Network class
class Network:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_node(self, name, params):
        """
        Add a genetic component (e.g., gene, promoter, RBS) to the network.
        Biological Meaning:
        - Nodes represent genetic parts or molecules.
        - Parameters include promoter strength, RBS efficiency, etc.
        """
        self.graph.add_node(name, **params)
    
    def add_edge(self, source, target, weight):
        """
        Add regulatory interaction between components.
        Biological Meaning:
        - Edges represent interactions like activation or repression.
        - Weights can model interaction strengths (e.g., binding affinities).
        """
        self.graph.add_edge(source, target, weight=weight)

# Define Simulation class
class Simulation:
    def __init__(self, network, gate_type="OR"):
        """
        Initialize the simulation.
        :param network: Genetic circuit network.
        :param gate_type: Logic gate type ('OR' or 'AND').
        """
        self.network = network
        self.gate_type = gate_type
        
    def hill_function(self, x, K, n):
        """
        Hill function for cooperative binding.
        :param x: Input concentration (e.g., AHL or benzoate).
        :param K: Dissociation constant (binding affinity).
        :param n: Hill coefficient (cooperativity).
        :return: Fraction of bound molecules.
        """
        return x**n / (K**n + x**n)

    def simulate(self, params, t):
        """
        Simulate the genetic circuit based on its logic gate type.
        Biological Meaning:
        - Models how gene expression changes under the influence of AHL and benzoate.
        :param params: Parameter set (e.g., AHL, benzoate concentrations).
        :param t: Time array for simulation.
        """
        def odes(state, t, params):
            # State variables: state[0] = output gene expression
            AHL = params['AHL']  # AHL concentration
            benzoate = params['benzoate']  # Benzoate concentration

            if self.gate_type == "OR":
                # OR gate: Output is suppressed if AHL OR benzoate is high
                repression = AHL / (1 + AHL) + benzoate / (1 + benzoate)
            elif self.gate_type == "AND":
                # AND gate: Output is suppressed only if both AHL AND benzoate are high
                repression = (AHL / (1 + AHL)) * (benzoate / (1 + benzoate))
            else:
                raise ValueError("Unsupported gate type: {}".format(self.gate_type))

            # Hill function for output gene expression
            dydt = [-state[0] + (1 - repression)]  # Activation is 1 - repression
            return dydt

        # Initial state: Assume output is initially "off" (low expression)
        initial_state = [0.0]
        return odeint(odes, initial_state, t, args=(params,))
    
    @staticmethod
    def plot_hill_function(K, n, x_range=(0, 10), title="Hill Function"):
        """
        Visualize a Hill function.
        Args:
            K (float): Dissociation constant.
            n (float): Hill coefficient.
            x_range (tuple): Range of x values to plot.
            title (str): Title of the plot.
        """
        x = np.linspace(x_range[0], x_range[1], 500)
        y = x**n / (K**n + x**n)  # Hill function formula

        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Hill(K={K}, n={n})"))
        fig.update_layout(
            title=title,
            xaxis_title="Input (x)",
            yaxis_title="Output",
            template="plotly_dark",
        )
        fig.show()

# Define Optimizer class
class Optimizer:
    def __init__(self, network, simulation):
        """
        Optimizer for genetic circuits.
        :param network: Network representation of the genetic circuit.
        :param simulation: Simulation object to evaluate dynamics.
        """
        self.network = network
        self.simulation = simulation

    def optimize(self, generations, population_size, fitness_fn):
        """
        Optimize the genetic circuit parameters using a Genetic Algorithm (GA).
        Biological Meaning:
        - Mimics evolutionary processes to find the best-performing parameter sets.
        - Fitness function evaluates biological quality (e.g., stability, output).
        :param generations: Number of generations to run the GA.
        :param population_size: Number of individuals in the population.
        :param fitness_fn: Custom fitness function to evaluate parameters.
        :return: Best parameter set after optimization.
        """
        if fitness_fn is None:
            raise ValueError("A fitness function must be provided for optimization.")

        # Initialize a random population of parameter sets
        population = [self.random_params() for _ in range(population_size)]
        print("Population at start of generation:", population)

        for generation in range(generations):
            # Evaluate the fitness of each individual
            fitness_scores = []
            for ind in population:
                try:
                    score = fitness_fn(ind)
                    fitness_scores.append((score, ind))
                except KeyError as e:
                    print("Error evaluating individual:", ind, e) #fitness_scores = [(fitness_fn(ind), ind) for ind in population]
            if not fitness_scores:
                raise ValueError(f"Generation {generation + 1}: No valid fitness scores. Check fitness function and input.")

            # Sort by fitness score (first element of the tuple)
            fitness_scores = sorted(fitness_scores, key=lambda x: x[0], reverse=True)
            # Print generation stats (optional)
            print(f"Generation {generation + 1}: Best Fitness = {-fitness_scores[0][0]:.4f}")

            # Select top individuals and breed the next generation
            #top_individuals = [ind for _, ind in fitness_scores[:population_size // 2]]
            top_individuals = [ind for _, ind in fitness_scores[:max(1, population_size // 2)]]
            population = self.breed_population(top_individuals)

        # Return the best parameter set
        return max(population, key=fitness_fn)
    
    def random_params(self):
        """
        Generate a random parameter set.
        Biological Meaning:
        - Models biological variability in promoter strength, RBS efficiency, etc.
        :return: Dictionary of randomized parameters.
        """
        return {
            'AHL': random.uniform(0, 10),         # AHL concentration
            'benzoate': random.uniform(0, 10),   # Benzoate concentration
            'K': random.uniform(1, 10),          # Hill dissociation constant
            'n': random.uniform(1, 4),           # Hill coefficient
        }

    #def breed_population(self, top_individuals):
    #    """
    #    Create a new population by breeding the top-performing parameter sets.
    #    Biological Meaning:
    #    - Simulates genetic recombination and mutation to generate diversity.
    #    :param top_individuals: List of top-performing parameter sets.
    #    :return: New population for the next generation.
    #    """
    #    next_population = []
    #    for i in range(len(top_individuals) // 2):
    #        parent1 = top_individuals[2 * i]
    #        parent2 = top_individuals[2 * i + 1]
    #        child = self.crossover(parent1, parent2)
    #        child = self.mutate(child)  # Introduce mutations
    #        # Ensure child has all required keys
    #        required_keys = ['AHL', 'benzoate', 'K', 'n']
    #        for key in required_keys:
    #            if key not in child:
    #                child[key] = random.uniform(0, 10) if key in ['AHL', 'benzoate'] else random.uniform(1, 4)
    #        
    #        next_population.append(child)
    #    return next_population
    
    def breed_population(self, top_individuals):
        """
        Create a new population by breeding the top-performing parameter sets.
        Ensures that the new population has the required size.
        """
        next_population = []

        # Breeding loop
        while len(next_population) < len(top_individuals) * 2:  # Adjust based on desired population growth
            parent1 = random.choice(top_individuals)
            parent2 = random.choice(top_individuals)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)  # Introduce mutations
            next_population.append(child)

        # Replenish population with random individuals if needed
        while len(next_population) < len(top_individuals) * 2:  # Adjust based on final population size needed
            next_population.append(self.random_params())

        return next_population
    
    def crossover(self, parent1, parent2):
        """
        Combine two parameter sets to create a child.
        Biological Meaning:
        - Models genetic recombination where offspring inherit traits from both parents.
        :param parent1: Parameter set of parent 1.
        :param parent2: Parameter set of parent 2.
        :return: Combined parameter set (child).
        """
        return {k: (parent1[k] + parent2[k]) / 2 for k in parent1.keys()}

    def mutate(self, params):
        """
        Introduce random changes (mutations) to a parameter set.
        Biological Meaning:
        - Mimics DNA mutations that introduce diversity into the population.
        :param params: Original parameter set.
        :return: Mutated parameter set.
        """
        mutation_rate = 0.1  # Percentage change allowed
        return {k: v + random.uniform(-mutation_rate * v, mutation_rate * v) for k, v in params.items()}
