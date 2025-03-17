import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter




'''
each of eta, Gamma, Delta are arrays of length 2 (nothign coupling between each of 0 and 1 states to 2 state)

Below is for computing ss without using the effective operator formalism
'''
def compute_ss(n_motional,nu,Delta,Gamma,eta): 

    # Convention: |g⟩⟨g| and |e⟩⟨e| for ground and excited states
    zero_proj = qt.basis(3, 0).proj()
    one_proj = qt.basis(3, 1).proj()
    
    ground_proj = zero_proj + one_proj
    excited_proj = qt.basis(3, 2).proj()


    D0 = qt.basis(3,0) * qt.basis(3,2).dag()
    D1 = qt.basis(3,1) * qt.basis(3,2).dag()

    # motional state operators
    a = qt.destroy(n_motional)     # Annihilation operator
    a_dag = qt.create(n_motional)  # Creation operator

    # Position operator x 
    x = (a + a_dag) # I ignore units here because we just roll with eta as a parameter rather than worrying about 'k'

    # Identity operators for tensor products
    id_internal = qt.qeye(3)
    id_motional = qt.qeye(n_motional)

    # Define e^(±ikx) operators using the position operator
    # We use the exponential series expansion
    def exp_ikx(eta):
        exp_x = (1j * eta * x).expm() # I checked that expm works.
        return exp_x

    #Note that we follow a convention - the order is always atomic tensor motional

    '''
    below I introduce the 3LS operators operators
    
    '''
    Gamma0 = Gamma[0]
    Gamma1 = Gamma[1]

    eta0 = eta[0]
    print(eta0)
    exit(0)
    eta1 = eta[1]

    Delta0 = Delta[0]
    Delta1 = Delta[1]
    
    Omega0 = eta0*Gamma0
    Omega1 = eta1*Gamma1

    print(Gamma0, eta0, Delta0, Omega0)
    print(Gamma1, eta1, Delta1, Omega1) 
    exit(0)

    H_motional = nu * qt.tensor(id_internal, a_dag * a)
    H_atomic = qt.tensor(-Delta0 * qt.fock_dm(3,0) - Delta1* qt.fock_dm(3,1) , id_motional)       #  a |0><0| +  b |1><1|, set energy of |e><e| = 0 
    H_interaction = (
    Omega0/2* qt.tensor(D0 , exp_ikx(-eta0)) + Omega0/2* qt.tensor(D0.dag() , exp_ikx(eta0)) +   # |0><e| + c.c
    Omega1/2* qt.tensor(D1 , exp_ikx(-eta1)) + Omega1/2* qt.tensor(D1.dag() , exp_ikx(eta1))     # |0><e| + c.c
    )   

    H_total = H_motional + H_atomic + H_interaction

    # defining collapse operators
    # collapse operators for emission in positive and negative x-direction (1D)
    L0_plus = np.sqrt(Gamma0) * qt.tensor(D0,  exp_ikx(eta0))
    L0_minus = np.sqrt(Gamma0) * qt.tensor(D0, exp_ikx(-eta0))
    L1_plus = np.sqrt(Gamma1) * qt.tensor(D1,  exp_ikx(eta1))
    L1_minus = np.sqrt(Gamma1) * qt.tensor(D1, exp_ikx(-eta1))
    c_ops = [L0_plus, L0_minus, L1_plus, L1_minus]


    '''
    Rather than using MESolver we try using SSSolver (only interested in long-term behavior of system) - as done in the paper. 

    Get steady state values of the system as a function of diff parameters. 
    '''


    rho_ss = qt.steadystate(H_total, c_ops)

    # Measurement operators
    measure_excited = qt.tensor(excited_proj, id_motional)
    measure_ground = qt.tensor(ground_proj, id_motional)
    measure_n = qt.tensor(id_internal, a_dag * a)

    # Compute steady-state expectation values
    excited_pop_ss = qt.expect(measure_excited, rho_ss)
    ground_pop_ss = qt.expect(measure_ground, rho_ss)
    motional_n_ss = qt.expect(measure_n, rho_ss)


    return ground_pop_ss, excited_pop_ss, motional_n_ss





# example system parameters based on physical grounds
n_motional = 5         # Number of motional levels
Gamma = np.array([1e6 , 1e6]) # Spontaneous emission rate (Γ) in Hz
Delta = 0.5 * Gamma  # Detuning (Δ) in Hz
nu = 0.5 * Gamma[0]# Trap frequency (ν) in Hz 

eta = np.array([0.05, 0.05])              #Lamb-dicke parameter
# Eta removes the necessity for defining k because already given

Omega = eta*Gamma 
ground_pop_ss, excited_pop_ss, motional_n_ss = compute_ss(n_motional,nu,Delta,Gamma,eta)


print("Steady-state excited population:", excited_pop_ss)
print("Steady-state ground population:", ground_pop_ss)
print("Steady-state average phonon number:", motional_n_ss)

# for our upcoming plots, we can take freedom in varying nu, Delta for a given Gamma (of 0.01 MHz)