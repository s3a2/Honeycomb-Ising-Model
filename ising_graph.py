"""
Glauber Dynamics Ising Model
Graph Implementation

By:
- Shaan Sidhwani
- Blake Steadham

Last Modified 12/10/24 
"""


import numpy as np
import matplotlib.pyplot as plt


class Vertex():
    """
    Defines individual spin vertices within an Ising Model graph.

    Args:
        label (str): Label assigned to the particle should one choose to search for
        a given lattice point when performing operations such as adding graph edges.
        spin (int): The discrete direction of the magnetic moment for a particle
        at a particular lattice site. Either +1 (up) or -1 (down).
    """

    def __init__ (self, label=None, spin= -1):
        """Initializes vertex with class parameters. Also creates list to track
        neighboring vertices for iteration purposes."""
        self.label = label
        self.spin = spin
        self.neighbors = []


    def get_label(self):
        """Obtains the already-assigned label of a vertex."""
        return self.label


    def __str__(self):
        """Returns string representation of vertex as just its label."""
        return str(self.label)



class IsingGraph():
    """
    Defines an Ising Model graph that houses individual spin vertices of the Vertex()
    class defined above. This class can be used to represent different kinds of 
    Ising Model lattices or even absurd pathological cases.

    Args:
        coupling (float): The coupling strength with which lattice sites influence the
        spins of their neighbors. A higher coupling encourages the probability of spin
        allignment amongst neighbors.

        temp (float): The physical temperature of the system in arbitrary units. The
        higher the temperature, the more noise added to spin flipping probabilities
        as a consequence of thermal excitations.

        algorithm (str): Either "Glauber" or "Metropolis." Selects which algorithm will
        be employed to facilitate Ising model simulation.

        store_history (boolean): Determines whether or not to store history of calculating
        physical parameters and lattice states during simulation.
    """

    def __init__(self, coupling, temp, algorithm="Glauber", store_history=True):
        """
        Initializes Ising Model class with given class parameters.
        
        Additional Attributes:
        vertices (list): Keeps track of vertices (Vertex class objects) contained in graph.
        self.adj_mat (np array): 2D NumPy array containing graph adjacency matrix.
        self.lattice = (np array): Stores vertices (Vertex objects) in 2D array representing
        Ising graph.
        history, energy_history, magnetization_history (list): Lists used to store past
        lattices, total energy computations, and normalized magnetization computations.
        """
        self.vertices = []
        self.adj_mat = np.array([])
        self.coupling = coupling
        self.temp = max(temp, 1e-5)
        self.lattice = np.array([])
        self.algorithm = algorithm
        self.store_history = store_history
        self.history = []
        self.energy_history = []
        self.magnetization_history = []


    def has_vertex(self, label):
        """Checks if a vertex object with a given label is present in the graph."""
        for vertex in self.vertices:
            if label == vertex.get_label():
                return True
        return False


    def get_index(self, label):
        """Retrieves a vertex index given its label (if contained in graph). Else
        returns -1."""
        for i, vertex in enumerate(self.vertices):
            if label == vertex.get_label():
                return i
        return -1


    def add_vertex(self, label, spin):
        """Adds a vertex of given label to the graph if not already present."""
        if not self.has_vertex(label):
            self.vertices.append(Vertex(label, spin))

            if self.adj_mat.size == 0:
                self.adj_mat = np.zeros((1,1))
            resized_adj = np.zeros((len(self.vertices), len(self.vertices)))
            resized_adj[:self.adj_mat.shape[0], :self.adj_mat.shape[1]] = self.adj_mat
            self.adj_mat = resized_adj


    def dir_edge(self, start, finish, weight = 1):
        """Adds a weighted directed edge to the graph adjacency matrix."""
        vertex_labels = {vertex.label for vertex in self.vertices}

        if start in vertex_labels and finish in vertex_labels:
            self.adj_mat[start][finish] = weight


    def undir_edge(self, start, finish, weight = 1):
        """Adds a weighted undirected edge to the graph adjacency matrix."""
        vertex_labels = {vertex.label for vertex in self.vertices}

        if start in vertex_labels and finish in vertex_labels:
            self.adj_mat[start, finish] = weight
            self.adj_mat[finish, start] = weight


    def random_square(self, dim):
        """Populates a 2D square lattice graph with random spin vertices."""
        self.lattice = np.array([[Vertex(spin=np.random.choice([-1, 1]))
            for _ in range(dim)] for _ in range(dim)])

        self._square_adjacency()
        self._get_neighbors_square()
        return self.lattice


    def _get_neighbors_square(self):
        """Constructs a list of all vertices that neighbor a particular lattice point in
        the square lattice."""
        dim = self.lattice.shape[0]
        for i in range(dim):
            for j in range(dim):
                self.lattice[i][j].neighbors = [
                    self.lattice[(i+1)%dim, j],
                    self.lattice[(i-1)%dim, j],
                    self.lattice[i, (j+1)%dim],
                    self.lattice[i, (j-1)%dim]
                    ]


    def _square_adjacency(self):
        """Constructs and returns the adjacency matrix for a square lattice."""
        dim = self.lattice.shape[0]
        self.adj_mat = np.zeros((dim**2, dim**2))
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[0]):
                self.adj_mat[i*dim+j, ((i+1)%dim)*dim+j] = 1
                self.adj_mat[i*dim+j, i*dim+(j+1)%dim] = 1

        self.adj_mat += self.adj_mat.T
        return self.adj_mat


    def get_square_energy(self):
        """
        Calculates the total energy of a 2D Ising model square lattice
        using a Neumann neighborhood. Returns the total energy of the lattice
        expressed as a float.
        """

        total_energy = 0
        dim = self.lattice.shape[0]
        for i in range(dim):
            for j in range(dim):
                spin = self.lattice[i, j].spin
                for neighbor in self.lattice[i, j].neighbors:
                    step_energy = -self.coupling * spin * neighbor.spin
                    total_energy -= step_energy
        return total_energy / 2


    def honeycomb(self, dim):
        """Populates a 2D honeycomb lattice graph with random spin vertices."""
        dim += dim%2
        self.lattice = np.zeros((dim, dim))

        for i in range(dim):
            start = 0 if i % 2 == 0 else 2
            self.lattice[i, start::4] = Vertex(spin=np.random.choice([Vertex(spin=-1),
                Vertex(spin=1)]))
            self.lattice[i, start+1::4] = Vertex(spin=np.random.choice([Vertex(spin=-1),
                Vertex(spin=1)]))

        for i in range(1, dim-1):
            for j in range(1, dim-1):
                self.lattice[i][j].neighbors = self.find_nonzero_neighbors(i, j)

        return self.lattice


    def get_honeycomb_energy(self):
        """We elected to investigate honeycomb energy in a separate ipynb.
        One future direction could be to check if we obtain redundant 
        results in a graph implementation."""
        raise NotImplementedError


    def find_nonzero_neighbors(self, i, j):
        """Returns a list of nonzero lattice points in a 3x3 subarray surounding
        a particular lattice point. Can be used to determine the neighborhood
        of graphs represented by 2D arrays such as for the honeycomb lattice."""
        subarray = self.lattice[i-1:i+2, j-1:j+2]
        nonzero_vertices = np.argwhere(subarray != 0)
        return [(i-1 + v[0], j-1 + v[1]) for v in nonzero_vertices
            if (v[0], v[1]) != (1, 1)]


    def _get_site_energy(self, i, j):
        """Determines the change in energy of flipping the spin at a particular
        site. Used in determining spin flip probability."""
        spin_interaction_energy = self.lattice[i,j].spin * sum([vertex.spin
            for vertex in self.lattice[i,j].neighbors])
        return -2*self.coupling * spin_interaction_energy


    def get_norm_magnetization(self):
        """Obtains the normalized magnetization for an entire Ising
        graph or lattice."""
        magnetization = sum([vertex.spin for _ in self.lattice
            for vertex in _ if vertex != 0])

        num_lattice_pts = len([None for _ in self.lattice
            for vertex in _ if vertex != 0])

        return magnetization / num_lattice_pts if num_lattice_pts != 0 else None


    def _step(self):
        """One iteration of modifying the Ising model lattice. Chooses a
        random lattice site and determines if it flips according to the
        change in energy a spin flip would produce."""
        dim = self.lattice.shape[0]
        r = np.random.randint(dim)
        c = np.random.randint(dim)
        energy_change = self._get_site_energy(r, c)

        if self.algorithm.lower() == "glauber":
            p = 1 / (1 + np.exp(energy_change / self.temp))
        elif self.algorithm.lower() == "metropolis":
            p = min(1, np.exp(-energy_change / self.temp))
        else:
            raise ValueError("Invalid algorithm.")

        if np.random.rand() < p:
            self.lattice[r, c].spin *= -1


    def run(self, n_steps, lattice_type="square"):
        """
        Runs the Ising model for a specified lattice type and number of
        steps.

        Args:
            n_steps (int): Determines the number of iteration steps to run the Ising
            Model for.
            lattice_type (str): Determines the type of lattice to use when calculating
            total energy of Ising lattice.
        """
        for _ in range(n_steps):
            self._step()
            if self.store_history is True and not np.array_equal(self.history[-1],self.lattice):
                self.history.append(self.lattice.copy())
            if lattice_type.lower().strip() == "square":
                self.energy_history.append(self.get_square_energy())
            if lattice_type.lower().strip() == "honeycomb":
                self.energy_history.append(self.get_honeycomb_energy())
            self.magnetization_history.append(self.get_norm_magnetization())


    def __str__(self):
        """String representation of the adjacency matrix."""
        return np.array_str(self.adj_mat)



##### MAIN PROGRAM #####

def simulate(n_steps=100, algorithm="Glauber", y_var="energy", store_history=False,
    coupling=2.0, temp=10, lattice="square", dim=50):
    """Fully simulates and graphs Ising model physical parameters for a
    specified number of iterations.
    
    Args:
        n_steps (int): Determines the number of iteration steps to run the Ising
        Model for.
        algorithm (str): Either "Glauber" or "Metropolis." Specifies algorithm
        with which to calculate spin flip probabilities.
        y_var (str): Either "energy" or "magnetization." String that specifies
        which physical variable, either the total energy of the lattice or
        normalized lattice magnetization, will appear on the y-axis of the graph.
        store_history (boolean): Determines whether or not to store history of
        calculating physical parameters and lattice states during simulation.
        coupling (float): The coupling strength with which lattice sites influence the
        spins of their neighbors. A higher coupling encourages the probability of spin
        allignment amongst neighbors.
        temp (float): The physical temperature of the system in arbitrary units. The
        higher the temperature, the more noise added to spin flipping probabilities
        as a consequence of thermal excitations.
        lattice (str): String that specifies which type of lattice, "square" or
        "honeycomb," to compute the total energy for.
        dim (int): Integer that speicifies the dimension of the 2D array used to
        represent the Ising graph.
    """

    plot_title = ""
    if y_var.lower().strip() == "magnetization":
        plot_title += "Normalized Magnetization vs. Iterations (J="
    elif y_var.lower().strip() == "energy":
        plot_title += "Lattice Energy vs. Iterations (J="
    plot_title += str(coupling) + ", T=" + str(temp) + ")"

    ising = IsingGraph(coupling, temp, algorithm, store_history)
    ising.random_square(dim)
    ising.run(n_steps, lattice)
    plt.figure(figsize=(10, 6))
    if y_var.lower().strip() == "magnetization":
        plt.plot(range(n_steps), ising.magnetization_history, color='blue')
    elif y_var.lower().strip() == "energy":
        plt.plot(range(n_steps), ising.energy_history, color='orange', label ='J=2.0')

    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.title(plot_title, fontsize=16)

    plt.legend()
    plt.grid()
    plt.show()


def main():
    """Executes main program functionality of simulating the ising model and 
    graphing results. One could produce either a normalized magnetization vs.
    iterations or total energy vs. iterations graph. One future direction
    is implementing support for honeycomb lattices to identify whether or not
    we obtain redundant results to our other honeycomb lattice implementation."""
    simulate(n_steps=100, algorithm="Glauber", y_var="energy", coupling=2.0,
        temp=1e-5, lattice="square", dim=10)

if __name__ == '__main__':
    main()

