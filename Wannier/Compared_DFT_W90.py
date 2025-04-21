# import matplotlib as mpl
# # mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import re

"""
This code can compared band structure between DFT-VASP and Wannier90.

VASP data: BAND.data, KLABELS
We need to do pre- and post-process by vaspkit to extract data above.

Wannier90: wannier90_band.dat

We also need to alter the fermi level in the code to alighn the band structure, usually on the valence band minimum.
The inital value can be otained from the VASP OUTCAR file.
"""

E_fermi = 2.3

# Get the current file path and change it
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

def filter_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    filtered_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    return filtered_lines

def split_numbers(lines):
    split_lines = [list(map(float, line.split())) for line in lines]
    return split_lines

# Post-process file from vaspkit, 'vaspkit -task 211'

file_path = 'BAND.dat'
filtered_lines = filter_lines(file_path)
# print(filtered_lines)
split_lines = split_numbers(filtered_lines)

data = np.array(split_lines)

print(data[0])

def extract_nkpts_nbands(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    second_line = lines[1]
    # Use regular expressions to find the values of NKPTS and NBANDS
    match = re.search(r'NKPTS\s*&\s*NBANDS:\s*(\d+)\s+(\d+)', second_line)
    if match:
        nkpt = int(match.group(1))
        nband = int(match.group(2))
        return nkpt, nband
    else:
        raise ValueError("Could not find NKPTS and NBANDS values in the second line, please check the file format in BAND.dat created by vaspkit code.")

nkpt, nband = extract_nkpts_nbands(file_path)
# load data
import numpy as np

def extract_labels_and_numbers(file_path):
    labels_and_numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                break
            match = re.search(r'([^\d]+)(\d+\.\d+|\d+)', line)
            if match:
                label = match.group(1).strip()
                number = match.group(2)
                labels_and_numbers.append((label, number))
    return labels_and_numbers

# We also need pre-process from vaspkit to obain the recommended k-path
file_path = 'KLABELS'
labels_and_numbers = extract_labels_and_numbers(file_path)

high_k =[]
label_k = []
for label, number in labels_and_numbers:
    label_k.append(label)
    high_k.append(float(number))
label_k = ['\u0393' if x == 'GAMMA' else x for x in label_k]

print(high_k)
print(len(data))

#orbit = ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2']

plt.rcParams["figure.dpi"]= 300
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["figure.figsize"]=(4.3, 3.2)
plt.rcParams["font.family"]="Arial"

bands = np.reshape(data[:, 1], (-1, nkpt))
k = np.reshape(data[:, 0], (-1, nkpt))
    
for band in range(bands.shape[0]):
    plt.plot(k[band,:], bands[band, :]-E_fermi, linewidth=1, alpha=1, color='k')
    for i in high_k[1:-1]:
        plt.axvline(i, linewidth=0.5, linestyle = '--',color='#444444', alpha=0.5)


def read_wannier90_band(file_path):
    with open(file_path, 'r') as file:
        data = []
        segment = []
        for line in file:
            if line.strip():  # skip empty lines
                segment.append([float(x) for x in line.split()])
            else:
                if segment:
                    data.append(segment)
                    segment = []
        if segment:
            data.append(segment)
    return data

# load wannier90_band.da
file_path = 'wannier90_band.dat'
data_segments = read_wannier90_band(file_path)

# extract k-points and bands, the k_path data are repeated
k_points = np.array(data_segments[0])[:, 0]  
bands1 = np.concatenate([np.array(segment)[:, 1] for segment in data_segments], axis=0)

nkpt = len(k_points)
bands2 = np.reshape(bands1, (-1, nkpt))
for band in range(bands2.shape[0]):
    plt.plot(k_points, bands2[band, :] - E_fermi, linewidth=1, alpha=1, color='r')
    # for i in high_k[1:-1]:
    #     plt.axvline(i, linewidth=0.5, linestyle='--', color='#110000', alpha=0.5)


plt.xlim(min(k[0]), max(k[0]))
plt.ylim(-6, 6)
# Add legends
plt.plot([], [], color='k', label='DFT')
plt.plot([], [], color='r', label='Wannier')
plt.legend(fontsize=12, loc='upper right', frameon=False, shadow=False, fancybox=False, bbox_to_anchor=(1, 1), borderaxespad=0.5)
# text labels
plt.xticks(ticks= high_k, \
        labels=label_k,fontdict={'family':'arial','size':14})

plt.ylabel("Energy (eV)",fontdict={'family':'arial','size':14})  
# plt.show()
plt.savefig('band.svg',dpi=300)
