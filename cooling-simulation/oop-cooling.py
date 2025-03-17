import numpy as np
import qutip as qt


class TwoLevelSystem:
    """A class representing a two-level quantum system with motional degrees of freedom."""
    
    def __init__(self, n_motional=5, nu=2.0, Delta=-0.5, Omega=0.1, Gamma=0.05, k=1.0):
        """
        Initialize the quantum system with given parameters.
        
        Parameters:
        -----------
        n_motional : int
            Number of motional levels to consider
        nu : float
            Trap frequency in MHz (will be converted to angular frequency)
        Delta : float
            Detuning in MHz (will be converted to angular frequency)
        Omega : float
            Rabi frequency in MHz (will be converted to angular frequency)
        Gamma : float
            Spontaneous emission rate in MHz (will be converted to angular frequency)
        k : float
            Wavevector (dimensionless)
        """
        # Convert frequencies to angular frequencies
        self.n_motional = n_motional
        self.nu = nu * 2 * np.pi
        self.Delta = Delta * 2 * np.pi
        self.Omega = Omega * 2 * np.pi
        self.Gamma = Gamma * 2 * np.pi
        self.k = k
        
        # Initialize operators
        self._init_operators()
        
    def _init_operators(self):
        """Initialize all quantum operators for the system."""
        # Internal state operators
        self.ground_proj = qt.basis(2, 0).proj()
        self.excited_proj = qt.basis(2, 1).proj()
        self.D = qt.sigmam()        # Dipole lowering operator
        self.D_dag = qt.sigmap()    # Dipole raising operator
        
        # Motional state operators
        self.a = qt.destroy(self.n_motional)
        self.a_dag = qt.create(self.n_motional)
        self.n_op = self.a_dag * self.a
        
        # Position operator
        self.x = (self.a + self.a_dag) / np.sqrt(2)
        
        # Identity operators
        self.id_internal = qt.qeye(2)
        self.id_motional = qt.qeye(self.n_motional)
    
    def exp_ikx(self, k_factor=1.0):
        """Calculate the exponential operator e^(i*k_factor*k*x)."""
        return (1j * k_factor * self.k * qt.tensor(self.id_internal, self.x)).expm()
    
    def build_hamiltonian(self):
        """Build the total Hamiltonian for the system."""
        # Trap Hamiltonian
        H_trap = self.nu * qt.tensor(self.id_internal, self.n_op)
        
        # Atomic Hamiltonian
        H_atomic = self.Delta * qt.tensor(self.excited_proj, self.id_motional)
        
        # Interaction Hamiltonian
        H_int = -self.Omega/2 * (self.exp_ikx(-1) * qt.tensor(self.D_dag, self.id_motional) + 
                               self.exp_ikx(1) * qt.tensor(self.D, self.id_motional))
        
        # Total Hamiltonian
        self.H = H_trap + H_atomic + H_int
        return self.H
    
    def build_collapse_operators(self):
        """Build the collapse operators for the system."""
        # Emission in positive x-direction
        L_plus = np.sqrt(self.Gamma) * self.exp_ikx(1) * qt.tensor(self.D, self.id_motional)
        
        # Emission in negative x-direction
        L_minus = np.sqrt(self.Gamma) * self.exp_ikx(-1) * qt.tensor(self.D, self.id_motional)
        
        self.c_ops = [L_plus, L_minus]
        return self.c_ops
    
    def initial_state(self, internal_state=0, motional_state=3):
        """
        Create an initial state for the system.
        
        Parameters:
        -----------
        internal_state : int
            0 for ground state, 1 for excited state
        motional_state : int
            Fock state of the motional mode
        """
        return qt.tensor(qt.basis(2, internal_state), qt.basis(self.n_motional, motional_state))
    
    def get_measurement_operators(self):
        """Define common measurement operators."""
        measure_excited = qt.tensor(self.excited_proj, self.id_motional)
        measure_ground = qt.tensor(self.ground_proj, self.id_motional)
        measure_n = qt.tensor(self.id_internal, self.n_op)
        
        return {
            'excited': measure_excited,
            'ground': measure_ground,
            'phonon_number': measure_n
        }
    
    def run_simulation(self, tlist, initial_state=None):
        """
        Run the quantum simulation.
        
        Parameters:
        -----------
        tlist : array
            List of times to evaluate the system at
        initial_state : Qobj, optional
            Initial state of the system. If None, uses excited state with motional ground state.
        
        Returns:
        --------
        result : Result
            QuTiP result object containing the time evolution
        """
        if initial_state is None:
            initial_state = self.initial_state(1, 0)
            
        # Make sure Hamiltonian and collapse operators are built
        H = self.build_hamiltonian()
        c_ops = self.build_collapse_operators()
        
        # Run the simulation
        result = qt.mesolve(H, initial_state, tlist, c_ops, [])
        return result
    
    def analyze_results(self, result):
        """
        Analyze the simulation results.
        
        Parameters:
        -----------
        result : Result
            QuTiP result object from the simulation
            
        Returns:
        --------
        data : dict
            Dictionary containing expectation values
        """
        operators = self.get_measurement_operators()
        
        data = {}
        for name, op in operators.items():
            data[name] = qt.expect(op, result.states)
            
        return data




if __name__ == "__main__":
    # Create the quantum system
    system = QuantumTwoLevelSystem(
        n_motional=10,
        nu=2.0,       # MHz
        Delta=-0.1,   # MHz
        Omega=0.1,    # MHz
        Gamma=0.1,   # MHz
        k=1.0
    )
    
    # Define simulation time points (in microseconds)
    times = np.linspace(0, 100, 10000)
    
    # Run the simulation with default initial state (excited state, motional ground state)
    result = system.run_simulation(times)
    
    # Analyze the results
    data = system.analyze_results(result)
    
    # Now data containtensorta['ground'], label='Ground State')
    plt.plot(times, data['phonon_number'], label='Phonon Number')
    plt.xlabel('Time (μs)')
    plt.ylabel('Population / Number')
    plt.legend()
    plt.title('Quantum System Dynamics')
    plt.grid(True)
    plt.show()