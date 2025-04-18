## 
## ATTENTION: 
## Ensure that the carrier concentration satisfies the degenerate state condition
#  Example minimal threshold for degenerate state in  1e19 cm^-3


import math
import numpy as np
import os
import matplotlib.pyplot as plt

CBM = 1
VBM = -1
k_b = 1.381e-23 # Boltzman Constant
eV  = 1.600e-19 # Translate Joule to eletron voltage
Temp = 300 # K
volume = 60.25/10**24 # cm^-3
carrier_criteria = 1e15
loop_criteria = 100
band_gap = 0.5
Del_E = 0
k_T = k_b * Temp / eV # Boltzmann constant in eV/K

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'ZrCoSb.dat')

#  pass-test
def read_dos(path):
    global Del_E
    with open(path) as dos_file:
        dos = []
        energy = []
        next(dos_file)  # Skip the first line
        while True:
            lines = dos_file.readline()
            if not lines:
                break
            En_temp, DOS_temp = [float(i) for i in lines.split()]
            energy.append(En_temp)
            dos.append(DOS_temp)
        Del_E = energy[1] - energy[0]
        energy = np.array(energy)
        dos = np.array(dos)*2
    
    return energy, dos

#  pass-test
def find_band_edge(energy, dos):
    """
    This function is designed to identify and extract the valence band maximum (VBM) 
    and conduction band minimum (CBM) from the provided energy and density of states (DOS) arrays. 
    It also plots the corresponding DOS for the VBM and CBM regions.
        - vbm_dos (numpy.ndarray): The DOS values corresponding to the VBM region.
        - vbm_energy (numpy.ndarray): The energy values corresponding to the VBM region.
        - cbm_dos (numpy.ndarray): The DOS values corresponding to the CBM region.
        - cbm_energy (numpy.ndarray): The energy values corresponding to the CBM region.
        - The function includes a plot of the DOS for the VBM and CBM regions, with fixed x-axis 
          and y-axis limits for visualization purposes.
    Usage:
        This function is primarily used for testing and visualizing the separation of 
        the density of states (DOS) corresponding to the valence band maximum (VBM) 
        and conduction band minimum (CBM).
    Parameters:
        energy (numpy.ndarray): A 1D array of energy values (in eV).
        dos (numpy.ndarray): A 1D array of density of states corresponding to the energy values.
    Returns:
        tuple: A tuple containing:
            - VBM (float): The energy value of the valence band maximum.
            - CBM (float): The energy value of the conduction band minimum.
    Notes:
        - The function assumes that the Fermi level is at 0.0 eV.
        - It identifies the VBM as the highest energy below the Fermi level with non-zero DOS.
        - It identifies the CBM as the lowest energy above the Fermi level with non-zero DOS.
    """
    global VBM
    global CBM
    start_point = np.where(energy <= 0.0, energy, -np.inf).argmax() # https://www.cnpython.com/qa/170441
    
    count = 0
    while dos[start_point - count] <= 1e-5:
        count = count + 1
    VBM = energy[start_point - count + 1]
    vbm_energy = energy[0:start_point - count + 1]
    vbm_dos = dos[0:start_point - count + 1]

    count = 1
    while dos[start_point + count] <= 1e-5:
        count = count + 1
    CBM = energy[start_point + count-1]
    cbm_energy = energy[start_point + count-1:]
    cbm_dos = dos[start_point + count-1:]
    
    
    # plt.plot(vbm_energy, vbm_dos)
    # plt.plot(cbm_energy, cbm_dos)
    # plt.xlim(-5, 55) 
    # plt.ylim(0, 10)
    # plt.show()    

    return vbm_dos, vbm_energy, cbm_dos, cbm_energy

# TODO:
# def gap_scissor(energy, dos, new_gap):
#     global band_gap
#     global CBM
#     global VBM
#     start_point = np.where(energy < 0.0, energy, -np.inf).argmax() # https://www.cnpython.com/qa/170441

#     # below_fermi = 
#     old_gap = CBM - VBM
#     change_gap = new_gap - old_gap
#     CBM += change_gap
#     return

# TODO:
# def boltz_dist(nsize, volume, eform, temp):
#     return (nsize / volume) * np.exp(-eform/ (k_b*temp/eV))

# out-of-use
def fermi_dirac_distribution(efermi: float, energy: np.ndarray) -> np.ndarray:
    """
    Calculate the Fermi-Dirac distribution for a given energy level using numpy.

    Parameters:
    -----------
    efermi : float
        The Fermi energy level.
    energy : numpy.ndarray
        The energy values for which the distribution is calculated.

    Returns:
    --------
    numpy.ndarray
        The Fermi-Dirac distribution values for the given energy levels.
    """
    kT = k_b * Temp / eV
    return 1 / (np.exp((energy - efermi) / kT) + 1)


def fd_intergal(efermi: float, energy: np.ndarray, dos: np.ndarray) -> float:
    """
    Calculate the Fermi-Dirac integral multipy? DOS? using the trapezoidal rule.
    ATTENTION !!!!
    1) This result does not include the volume, just actived fraction.
    2) ...
    Parameters:
    -----------
    efermi : float
        The Fermi energy level.
    energy : numpy.ndarray
        The energy values corresponding to the density of states.
    dos : numpy.ndarray
        The density of states values corresponding to the energy levels.

    Returns:
    --------
    The calculated Fermi-Dirac integral.
    P and N type Carrier density: float 
        
    """
    kT = k_b * Temp / eV
    # exp_arg = np.clip((energy - efermi) / kT, -1e10, 1e10)  # Avoid overflow in exp
    fd_value = 1 / (np.exp((energy-efermi)/ kT) + 1)
    # print(np.exp((energy-efermi)/ kT))
    n_type = np.trapezoid(dos * fd_value, energy)
    p_type = np.trapezoid(dos * (1 - fd_value), energy)
    # plt.plot(energy, dos, label='DOS')
    # plt.plot(energy, fd_value, label='Fermi-Dirac Distribution')
    # # plt.plot(energy, dos * fd_value,linestyle='--', label='n_type Carrier Density')
    # plt.legend()
    # plt.show()
    
    return p_type, n_type

def carrier_density(vbm_energy, vbm_dos, cbm_energy, cbm_dos,\
                    E_fermi):
    """
    ATTENTION !!!!
    1) This result contain the bipolor effect.
    2) ...
    """ 
    p_type, _ = fd_intergal(E_fermi, vbm_energy, vbm_dos)
    _, n_type = fd_intergal(E_fermi, cbm_energy, cbm_dos)
    total_density = (p_type + n_type)/volume
    return total_density  

def self_consist(vbm_energy, vbm_dos, cbm_energy, cbm_dos,\
                  E_fermi=(VBM+CBM)/2, lp=VBM-1, rp=CBM+1, \
                  count= 0, target_density=3.8e21):
    """
    parameters
    	E_fermi: initital point: in the middle between VBM and CBM
    	lp: left point
    	rp: right point

    """
    # vbm_range
    # cbm_range
    # total_charge = fermi_dist(vbm_range) + fermi_dist(cbm_range) + target_densit1y
    p_type, _ = fd_intergal(E_fermi, vbm_energy, vbm_dos)
    _, n_type = fd_intergal(E_fermi, cbm_energy, cbm_dos)
    total_density = (p_type - n_type)/volume - target_density

    # print(p_type)
    # print(p_type/volume)
    print(f"P-type density: {(p_type)/volume:.2e} cm^-3")
    # print(f"P-type density: {total_density:.2e} cm^-3")
    # print(E_fermi)
    if (np.abs(total_density) <= carrier_criteria): 
        print(f"Congratulation! Up to the Criterial {total_density:.2e}")

        return E_fermi
    elif (count>loop_criteria):
        print(f"Check, End only up to Maximum step: {total_density:.2e}")
        return E_fermi
    elif total_density < 0:
        return self_consist(vbm_energy, vbm_dos, cbm_energy, cbm_dos, \
                            rp = E_fermi,lp = lp, E_fermi=(E_fermi + lp)/2, \
                            count = count+1)
    else:
        return self_consist(vbm_energy, vbm_dos, cbm_energy, cbm_dos, \
                            lp = E_fermi,rp = rp, E_fermi=(E_fermi + rp)/2, 
                            count = count+1)


def main():
    """
    Main function to execute the Fermi level and density calculations.
    """
    
    energy, dos = read_dos(path)
    vbm_dos, vbm_energy , cbm_dos, cbm_energy = find_band_edge(energy, dos)
    print(f"VBM: {VBM:.2f} eV")
    print(f"CBM: {CBM:.2f} eV") 
    """
    TEST: Efermi-vs-Density
    """
    # E_fermi = np.linspace(-3, 2, 1000)
    # density = np.zeros(E_fermi.shape)
    # for i in range(len(E_fermi)):
    #     density[i] = carrier_density(vbm_energy, vbm_dos, cbm_energy, cbm_dos,\
    #                                   E_fermi[i])
    # plt.scatter(E_fermi, np.log10(np.abs(density)))
    # plt.axhline(y=0, color='r', linestyle='--')
    # # plt.ylim(0, 1e20)
    # plt.show()

    """
    TEST:  Fermi_Dirac intergral
    """
    exp_arg = np.clip((energy - 0) / k_T, -1e10, 1e10)  # Avoid overflow in exp
    fd_value = 1 / (np.exp(exp_arg) + 1)
    fd_int = np.trapezoid(fd_value, energy)
    print(fd_int)
    
    print(carrier_density(vbm_energy, vbm_dos, cbm_energy, cbm_dos,\
                    E_fermi=-0))
    
    E_fermi = self_consist(vbm_energy, vbm_dos, cbm_energy, cbm_dos, \
                           E_fermi=(VBM+CBM)/2, rp= CBM+2, lp=VBM-2)
    print(f"Calculated Fermi Level: {E_fermi}")
    
if __name__ == "__main__":
    main()