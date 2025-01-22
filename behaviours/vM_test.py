# %%
import os
import sys
from pathlib import Path
sys.path.append("/mofem_install/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/tfel-4.0.0-jjcwdu6cbil5dzqzjhjekn3jdzo3e6gc/lib/python3.11/site-packages")
import numpy as np
import mtest
from pydantic import BaseModel

class StressTestLoadCase(BaseModel):
    SXX: dict[float, float] | None = None
    SYY: dict[float, float] | None = None
    SZZ: dict[float, float] | None = None
    
    def unpack(self):
        if self.SXX:
            m.setImposedStress("SXX", self.SXX)
        if self.SYY:
            m.setImposedStress("SYY", self.SYY)
        if self.SZZ:
            m.setImposedStress("SZZ", self.SZZ)

class StrainTestLoadCase(BaseModel):
    EXX: dict[float, float] | None = None
    EYY: dict[float, float] | None = None
    EZZ: dict[float, float] | None = None
    EXY: dict[float, float] | None = None
    EYZ: dict[float, float] | None = None
    EXZ: dict[float, float] | None = None
    
    def unpack(self):
        if self.EXX:
            m.setImposedStrain("EXX", self.EXX)
        if self.EYY:
            m.setImposedStrain("EYY", self.EYY)
        if self.EZZ:
            m.setImposedStrain("EZZ", self.EZZ)
        if self.EXY:
            m.setImposedStrain("EXY", self.EXY)
        if self.EYZ:
            m.setImposedStrain("EYZ", self.EYZ)
        if self.EXZ:
            m.setImposedStrain("EXZ", self.EXZ)

m = mtest.MTest()
mtest.setVerboseMode(mtest.VerboseLevel.VERBOSE_QUIET)
m.setMaximumNumberOfSubSteps(20)
m.setModellingHypothesis("Tridimensional")

# model = "vMDefault"
model = "vM"
lib_path = "/mofem_install/jupyter/thomas/mfront_interface/src/libBehaviour.so"

b = mtest.Behaviour('generic', lib_path, model,'Tridimensional')
print(f"Material Properties: {b.getMaterialPropertiesNames()}")
print(f"State variables: {b.getInternalStateVariablesNames()}")
m.setBehaviour("generic", lib_path, model)
E = 500
nu = 0.3
H = 2.5
# material has a uniaxial yield stress of sig_y
sig_y = 0.25
tau_oct_0 = np.sqrt(2/3) * sig_y
# Loading programme
tMax = 1.0  # s , total time
nTime = 200
ltime = np.linspace(0.0, tMax, nTime)

# Environment parameters
m.setExternalStateVariable("Temperature", 293.15)
# Material parameters
m.setMaterialProperty("YoungModulus", E)
m.setMaterialProperty("PoissonRatio", nu)
m.setMaterialProperty("HardeningSlope", H)
m.setMaterialProperty("YieldStrength", sig_y)

controls = ["stress", "strain"]
control = controls[1]
if control == "stress":
    cyclic_stress = StressTestLoadCase(
        SZZ = {0: 0, 0.25: 1.0, 0.5: -1.5, 0.75: 0, 1.0: 1.75},
        SXX = {0: 0, 1.0: 0},
        SYY = {0: 0, 1.0: 0},
    )
    triaxial_test = StressTestLoadCase(
        SZZ = {0: 0, 0.25: 50, 0.5: 200, 0.75: 150, 1: 200},
        SXX = {0: 0, 0.25: 50, 1: 50},
        SYY = {0: 0, 0.25: 50, 1: 50},
    )
    etlc_cyclic_uniaxial_test = StressTestLoadCase(
        SZZ = {0: 0, 0.33: -0.3, 0.66: 0.393, 1: -0.4795},
        SXX = {0: 0, 0.25: 0, 1: 0},
        SYY = {0: 0, 0.25: 0, 1: 0},
    )
    
    chosen = etlc_cyclic_uniaxial_test
    chosen.unpack()
if control == "strain":
    etlc_compression_constant_strain = StrainTestLoadCase(
        EZZ = {0: 0, 0.25: 0.01, 0.5: 0.0095, 0.75: 0.015, 1: 0.0145},
        EXX = {0: 0, 1: 0},
        EYY = {0: 0, 1: 0},
    )
    etlc_cyclic_constant_strain = StrainTestLoadCase(
        EZZ = {0: 0, 0.25: 0.01, 0.5:0, 0.75:-0.01, 1:0.015},
        EXX = {0: 0, 1: 0},
        EYY = {0: 0, 1: 0},
    )
    etlc_shear = StrainTestLoadCase(
        EXZ = {0: 0, 1: 0.015}
    )
    etlc_cyclic_strain = StrainTestLoadCase(
        EZZ = {0: 0, 0.5: 0.20, 1: -0.10},
        EXX = {0: 0, 0.5: -0.03, 1: 0.05},
        EYY = {0: 0, 0.5: -0.03, 1: 0.05},
    )
    etlc_cyclic_uniaxial_strain = StrainTestLoadCase(
        EZZ = {0: 0, 
            1/4: -0.001, 
            2/4: 0, 
            3/4: 0.001,
            1: -0.001,
        },
        # EXX = {0: 0, 0.5: -0.03, 1: 0.05},
        # EYY = {0: 0, 0.5: -0.03, 1: 0.05},
    )

    chosen = etlc_cyclic_uniaxial_strain
    chosen.unpack()

# %%
s = mtest.MTestCurrentState()
wk = mtest.MTestWorkSpace()
m.completeInitialisation()
m.initializeCurrentState(s)
m.initializeWorkSpace(wk)

# initialize output lists
sig_xx = []
sig_yy = []
sig_zz = []
sig_xy = []
sig_xz = []
sig_yz = []
e_xx = []
e_yy = []
e_zz = []
e_xy = []
e_xz = []
e_yz = []
e_e_xx = []
e_e_yy = []
e_e_zz = []
e_e_xy = []
e_e_xz = []
e_e_yz = [] 
extracting_pl = True
if extracting_pl:
    e_p_xx = []
    e_p_yy = []
    e_p_zz = []
    e_p_xy = []
    e_p_xz = []
    e_p_yz = []
e_p_eq = []
# run sim
for i in range(0, nTime - 1):
    # print(f"===========Loop {i}===========")
    if i == 0:
        sig_xx.append(s.s0[0])
        sig_yy.append(s.s0[1])
        sig_zz.append(s.s0[2])
        sig_xy.append(s.s0[3])
        sig_xz.append(s.s0[4])
        sig_yz.append(s.s0[5])
        e_xx.append(s.e0[0])
        e_yy.append(s.e0[1])
        e_zz.append(s.e0[2])
        e_xy.append(s.e0[3])
        e_xz.append(s.e0[4])
        e_yz.append(s.e0[5])
        eel = s.getInternalStateVariableValue("ElasticStrain")
        e_e_xx.append(eel[0])
        e_e_yy.append(eel[1])
        e_e_zz.append(eel[2])
        e_e_xy.append(eel[3])
        e_e_xz.append(eel[4])
        e_e_yz.append(eel[5])
        if extracting_pl:
            epl = s.getInternalStateVariableValue("PlasticStrain")
            e_p_xx.append(epl[0])
            e_p_yy.append(epl[1])
            e_p_zz.append(epl[2])
            e_p_xy.append(epl[3])
            e_p_xz.append(epl[4])
            e_p_yz.append(epl[5])
        epleq = s.getInternalStateVariableValue("EquivalentPlasticStrain")
        e_p_eq.append(epleq)
    try:
        m.execute(s, wk, ltime[i], ltime[i + 1])
    except Exception as e:
        print(e)
        break
    sig_xx.append(s.s1[0])
    sig_yy.append(s.s1[1])
    sig_zz.append(s.s1[2])
    sig_xy.append(s.s1[3])
    sig_xz.append(s.s1[4])
    sig_yz.append(s.s1[5])
    e_xx.append(s.e1[0])
    e_yy.append(s.e1[1])
    e_zz.append(s.e1[2])
    e_xy.append(s.e1[3])
    e_xz.append(s.e1[4])
    e_yz.append(s.e1[5])
    eel = s.getInternalStateVariableValue("ElasticStrain")
    e_e_xx.append(eel[0])
    e_e_yy.append(eel[1])
    e_e_zz.append(eel[2])
    e_e_xy.append(eel[3])
    e_e_xz.append(eel[4])
    e_e_yz.append(eel[5])
    if extracting_pl:
        epl = s.getInternalStateVariableValue("PlasticStrain")
        e_p_xx.append(epl[0])
        e_p_yy.append(epl[1])
        e_p_zz.append(epl[2])
        e_p_xy.append(epl[3])
        e_p_xz.append(epl[4])
        e_p_yz.append(epl[5])
    epleq = s.getInternalStateVariableValue("EquivalentPlasticStrain")
    e_p_eq.append(epleq)
   
sig_xx = np.array(sig_xx)   
sig_yy = np.array(sig_yy)   
sig_zz = np.array(sig_zz)   
sig_xy = np.array(sig_xy)   
sig_xz = np.array(sig_xz)   
sig_yz = np.array(sig_yz)
e_xx = np.array(e_xx)   
e_yy = np.array(e_yy)   
e_zz = np.array(e_zz)   
e_xy = np.array(e_xy)   
e_xz = np.array(e_xz)   
e_yz = np.array(e_yz) 
e_e_xx = np.array(e_e_xx)   
e_e_yy = np.array(e_e_yy)   
e_e_zz = np.array(e_e_zz)   
e_e_xy = np.array(e_e_xy)   
e_e_xz = np.array(e_e_xz)   
e_e_yz = np.array(e_e_yz)  
if extracting_pl:
    e_p_xx = np.array(e_p_xx)   
    e_p_yy = np.array(e_p_yy)   
    e_p_zz = np.array(e_p_zz)   
    e_p_xy = np.array(e_p_xy)   
    e_p_xz = np.array(e_p_xz)   
    e_p_yz = np.array(e_p_yz)  
e_p_eq = np.array(e_p_eq)  

# %%
# Function to calculate principal stresses and directions
def calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz):
    sig_1 = []
    sig_2 = []
    sig_3 = []

    for i in range(len(sig_xx)):
        # Create stress tensor
        stress_tensor = np.array([
            [sig_xx[i], sig_xy[i], sig_xz[i]],
            [sig_xy[i], sig_yy[i], sig_yz[i]],
            [sig_xz[i], sig_yz[i], sig_zz[i]]
        ])

        # Calculate principal stresses (eigenvalues)
        principal_stresses, _ = np.linalg.eigh(stress_tensor)
        principal_stresses = np.sort(principal_stresses)[::-1]  # Sort in descending order

        # Append principal stresses to respective lists
        sig_1.append(principal_stresses[0])
        sig_2.append(principal_stresses[1])
        sig_3.append(principal_stresses[2])

    # Convert lists to numpy arrays
    sig_1 = np.array(sig_1)
    sig_2 = np.array(sig_2)
    sig_3 = np.array(sig_3)

    return sig_1, sig_2, sig_3

def calculate_p(sig_1, sig_2, sig_3):
    return (sig_1 + sig_2 + sig_3) / 3

def calculate_J2(sig_1, sig_2, sig_3):
    J2_list = (1/6) * ((sig_1 - sig_2) ** 2 + (sig_2 - sig_3) ** 2 + (sig_3 - sig_1) ** 2)
    
    return J2_list

# Function to calculate volumetric strain and deviatoric strain
def calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz):
    volumetric_strain_list = []
    deviatoric_strain_list = []

    for i in range(len(e_xx)):
        # Volumetric strain is the trace of the strain tensor
        volumetric_strain = e_xx[i] + e_yy[i] + e_zz[i]
        volumetric_strain_list.append(volumetric_strain)

        # Deviatoric strain components
        e_mean = volumetric_strain / 3
        e_dev_xx = e_xx[i] - e_mean
        e_dev_yy = e_yy[i] - e_mean
        e_dev_zz = e_zz[i] - e_mean
        e_dev_xy = e_xy[i]
        e_dev_xz = e_xz[i]
        e_dev_yz = e_yz[i]

        # Deviatoric strain magnitude
        deviatoric_strain = np.sqrt(2/3 * (e_dev_xx**2 + e_dev_yy**2 + e_dev_zz**2) + 2 * (e_dev_xy**2 + e_dev_xz**2 + e_dev_yz**2))
        deviatoric_strain_list.append(deviatoric_strain)

    volumetric_strain_list = np.array(volumetric_strain_list)
    deviatoric_strain_list = np.array(deviatoric_strain_list)
    return volumetric_strain_list, deviatoric_strain_list

sig_1, sig_2, sig_3 = calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz)
e_1, e_2, e_3 = calculate_principal_stresses(e_xx, e_yy, e_zz, e_xy, e_yz, e_xz)
p = calculate_p(sig_1, sig_2, sig_3)
et = (e_1+e_2+e_3)/3
J_2 = calculate_J2(sig_1, sig_2, sig_3)
J  = np.sqrt(J_2)
tau_oct = np.sqrt(2 * J_2)
sig_eq = np.sqrt(3 * J_2)
sig_eq_signed = np.sign(et) * sig_eq
print(f"sig_1: {sig_1[-1]}")
print(f"sig_2: {sig_2[-1]}")
print(f"sig_3: {sig_3[-1]}")
print(f"sig_xx: {sig_xx[-1]}")
print(f"sig_yy: {sig_yy[-1]}")
print(f"sig_zz: {sig_zz[-1]}")
print(f"sig_xy: {sig_xy[-1]}")
print(f"sig_yz: {sig_yz[-1]}")
print(f"sig_xz: {sig_xz[-1]}")
print(f"e_xx: {e_xx[-1]}")
print(f"e_yy: {e_yy[-1]}")
print(f"e_zz: {e_zz[-1]}")
print(f"e_xy: {e_xy[-1]}")
print(f"e_yz: {e_yz[-1]}")
print(f"e_xz: {e_xz[-1]}")
e_v, e_d = calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)
e_e_v, e_e_d = calculate_volumetric_and_deviatoric_strain(e_e_xx, e_e_yy, e_e_zz, e_e_xy, e_e_xz, e_e_yz)
if extracting_pl:
    e_p_v, e_p_d = calculate_volumetric_and_deviatoric_strain(e_p_xx, e_p_yy, e_p_zz, e_p_xy, e_p_xz, e_p_yz)

# %%
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

matplotlib.rc('figure', figsize=(7, 6))
plt.rcParams['animation.ffmpeg_path'] ='/mofem_install/jupyter/thomas/ffmpeg-7.0.2-amd64-static/ffmpeg'
# Initialize axes for 3D plotting
def init_axes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

# Plot cylinder along the space diagonal
def plot_cylinder(ax, radius, start_height=0, end_height=100, color=None):
    diagonal_direction = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    height = np.linspace(start_height, end_height, 50)
    angle = np.linspace(0, 2 * np.pi, 100)
    Height, Angle = np.meshgrid(height, angle)

    # Define orthogonal vectors to the diagonal direction
    arbitrary_vector = np.array([0, 0, 1])
    if np.allclose(np.dot(arbitrary_vector, diagonal_direction), 0):
        arbitrary_vector = np.array([0, 1, 0])  # Choose a different vector if parallel

    orthogonal_vector_1 = np.cross(diagonal_direction, arbitrary_vector)
    orthogonal_vector_1 /= np.linalg.norm(orthogonal_vector_1)
    orthogonal_vector_2 = np.cross(diagonal_direction, orthogonal_vector_1)
    orthogonal_vector_2 /= np.linalg.norm(orthogonal_vector_2)

    X = (radius * np.cos(Angle) * orthogonal_vector_1[0] +
         radius * np.sin(Angle) * orthogonal_vector_2[0] +
         Height * diagonal_direction[0])
    Y = (radius * np.cos(Angle) * orthogonal_vector_1[1] +
         radius * np.sin(Angle) * orthogonal_vector_2[1] +
         Height * diagonal_direction[1])
    Z = (radius * np.cos(Angle) * orthogonal_vector_1[2] +
         radius * np.sin(Angle) * orthogonal_vector_2[2] +
         Height * diagonal_direction[2])

    ax.plot_surface(X, Y, Z, alpha=0.5, color=color if color else 'm')
    # ax.scatter(0, 0, 0, color='r', s=100)

# Plot stress history with classification based on tau_oct
def plot_stress_history(ax, sig_1, sig_2, sig_3, tau_oct, tau_oct_0, save_as: str =None):
    mask_elastic = tau_oct < tau_oct_0
    mask_plastic = tau_oct >= tau_oct_0
    if np.any(mask_elastic):
        ax.plot(sig_1[mask_elastic], sig_2[mask_elastic], sig_3[mask_elastic], color='b', label='Initial elastic region', linewidth=5)
    if np.any(mask_plastic):
        ax.plot(sig_1[mask_plastic], sig_2[mask_plastic], sig_3[mask_plastic], color='orange', label='Expansion of the elastic region', linewidth=5)

    vol_stress_value = (sig_1 + sig_2 + sig_3)
    diagonal_direction = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    vol_stress_x = vol_stress_value * diagonal_direction[0]
    vol_stress_y = vol_stress_value * diagonal_direction[1]
    vol_stress_z = vol_stress_value * diagonal_direction[2]
    ax.plot(vol_stress_x, vol_stress_y, vol_stress_z, color='r', linestyle='--', label='Space Diagonal')
    # ax.plot([vol_stress_x[-1], sig_1[-1]], [vol_stress_y[-1], sig_2[-1]], [vol_stress_z[-1], sig_3[-1]], color='g', linestyle='--', label='Deviatoric Stress')

# Plot metadata like labels and planes
def plot_meta(ax, elev, azim, title):
    ax.set_xlabel(r'$\sigma_1$')
    ax.set_ylabel(r'$\sigma_2$')
    ax.set_zlabel(r'$\sigma_3$')
    ax.set_title(title)

    # Plot planes and add arrowheads with labels
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    zlim = ax.get_zlim()
    text_fontsize = 10
    # y-plane
    ax.plot([0, 0], ylim, [0, 0], color='k', linestyle='--', alpha=0.5)
    ax.text(0, ylim[1] * 1.1, 0, r'$\sigma_2$', color='k', fontsize=text_fontsize, verticalalignment='center_baseline', horizontalalignment='center')

    # x-plane
    ax.plot(xlim, [0, 0], [0, 0], color='k', linestyle='--', alpha=0.5)
    ax.text(xlim[1] * 1.1, 0, 0, r'$\sigma_1$', color='k', fontsize=text_fontsize, verticalalignment='center_baseline', horizontalalignment='center')

    # z-plane
    ax.plot([0, 0], [0, 0], zlim, color='k', linestyle='--', alpha=0.5)
    ax.text(0, 0, zlim[1] * 1.1, r'$\sigma_3$', color='k', fontsize=text_fontsize, verticalalignment='center_baseline', horizontalalignment='center')

    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='lower center')

    ax.set_axis_off()
    plt.tight_layout()

# Plot functions for specific variables
def plot_stress_field(sig_1, sig_2, sig_3, tau_oct, tau_oct_0, elev, azim, title: str = "", save_as: str =None):
    # final_p = (sig_1[-1] + sig_2[-1] + sig_3[-1]) / np.sqrt(3)
    max_p = (max(sig_1) + max(sig_2) + max(sig_3)) / np.sqrt(3)
    max_tau_oct = max(tau_oct)
    fig, ax = init_axes()
    plot_cylinder(ax, radius=tau_oct_0, end_height=max_p, color='m')
    plot_cylinder(ax, radius=max_tau_oct, end_height=max_p, color='c')
    
    ax.plot([], [], [], color='m', label='Initial Yield Surface')  # Invisible handle for legend
    ax.plot([], [], [], color='c', label='Final Yield Surface')  # Invisible handle for legend
    
    plot_stress_history(ax, sig_1, sig_2, sig_3, tau_oct, tau_oct_0)
    plot_meta(ax, elev, azim, title)
    if save_as:
        filepath = os.path.join(PLOT_DIR, save_as)
        plt.savefig(filepath)
        return filepath


# %%
def plot_2d_with_quiver(x, y, xlabel, ylabel, title, color='b', scale=1, linestyle='-', label=None, plastic_cutoff=None, save_as: str = None):
    plt.figure()
    tolerance = 1e-6
    
    gradient_tolerance = 0.01
    gradients = np.gradient(y)

    start_idx = 0
    for i in range(len(y)-1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        
        plt.quiver(x[i], y[i], dx, dy, color=color, angles='xy', scale_units='xy', scale=1, zorder=10)

    plt.scatter(x, y, color=color, s=0.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True,zorder=0)
    
    if save_as:
        filepath = os.path.join(PLOT_DIR, save_as)
        plt.savefig(f"{filepath}.png")
        return save_as
    plt.show()


def plot_2d_with_animation(x, y, xlabel, ylabel, title, color='b', scale=1, linestyle='-', label=None, plastic_cutoff=None, save_as: str = None):
    fig, ax = plt.subplots()
    tolerance = 1e-6

    ax.scatter(x, y, color=color, s=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, zorder=0)

    quiver_segments = []

    def init():
        return quiver_segments

    def update(frame):
        # Plot one quiver at a time to animate
        i = frame
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        
        # Add a new quiver to the plot
        quiver = ax.quiver(x[i], y[i], dx, dy, color=color, angles='xy', scale_units='xy', scale=1, zorder=10)
        quiver_segments.append(quiver)

        return quiver_segments

    ani = FuncAnimation(fig, update, frames=len(x) - 1, init_func=init, blit=False, repeat=False, interval=100)

    # Optional: Save the animation as MP4
    if save_as:
        filepath = os.path.join(PLOT_DIR, save_as)
        FFwriter = animation.FFMpegWriter(fps=30)
        ani.save(f'{filepath}.mp4', writer = FFwriter)
        # ani.save(f"{filepath}.mp4", writer='ffmpeg', fps=30)  # Save as MP4 using FFmpeg

    plt.show()

def create_plot(data, x_label, y_label, title, save_as, handle="on"):
    linestyle = "-"
    fig, ax = plt.subplots()
    max_x, max_y = float('-inf'), float('-inf')
    for x, y, label, color, cutoff in data:
        if x is not None and y is not None:
            if cutoff:
                mask_elastic = abs(y) < abs(cutoff)
                mask_plastic = abs(y) >= abs(cutoff)
                plt.plot(x[mask_elastic], y[mask_elastic], linestyle=linestyle, color='b', label=f"label")
                plt.plot(x[mask_plastic], y[mask_plastic], linestyle=linestyle, color='orange', label=label)
            else:
                plt.plot(x, y, linestyle=linestyle, color=color, label=label)
                plt.scatter(x, y, color='r', marker='x', s=1)
            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
    
    # Add axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    ax.grid(False)
    
    if handle == "off":
        ax.legend(handlelength=0, handletextpad=0)

    if save_as:
        plt.savefig(save_as)
        return save_as
    
def plot_x_ys(x_array: list, y_arrays, labels: list, cutoffs=None, x_label="", y_label="", title="", save_as: str = None, handle="on"):
    data = []
    for i in range(len(y_arrays)):
        data.append((x_array, y_arrays[i], labels[i], 'g', None))
    return create_plot(data, x_label, y_label, title, save_as, handle)


def plot_J_vs_axial_strain(J, e_zz, save_as: str =None):
    return plot_2d_with_quiver(e_zz, J, 'Axial Strain $\epsilon_{zz}$', 'Deviatoric Stress J', 'J - Axial Strain', plastic_cutoff= (sig_y + H * e_p_eq)/np.sqrt(3), save_as=save_as)

def plot_tau_vs_e_zz(tau_oct, e_zz, save_as: str =None):
    return plot_2d_with_quiver(e_zz, tau_oct, 'Axial Strain $\epsilon_{zz}$', 'Octahedral Shear Stress $\\tau_{oct}$', '$\\tau_{oct}$ - Axial Strain', plastic_cutoff= tau_oct_0 + H * e_p_eq * np.sqrt(2)/np.sqrt(3), save_as=save_as)

def plot_sig_eq_vs_e_zz(sig_eq, e_zz, save_as: str =None):
    return plot_2d_with_quiver(e_zz, sig_eq, 'Axial Strain $\epsilon_{zz}$', 'Equivalent Stress $\sigma_{eq}$', '$\sigma_{eq}$ - Axial Strain', plastic_cutoff= (sig_y + H * e_p_eq),save_as=save_as)



def plot_volumetric_strain_vs_axial_strain(e_zz, e_v=None, e_e_v=None, e_p_v=None, save_as: str = None):
    data = [
        (e_zz, e_v, 'Total $\epsilon^{tot}_v$', 'g', None),
        (e_zz, e_e_v, 'Elastic $\epsilon^e_v$', 'b', None),
        (e_zz, e_p_v, 'Plastic $\epsilon^p_v$', 'r', None),
    ]
    return create_plot(data, 'Axial Strain $\epsilon_{zz}$', 'Volumetric Strain $\epsilon_v$', 'Volumetric Strain vs Axial Strain', save_as)

def plot_deviatoric_strain_vs_axial_strain(e_zz, e_d=None, e_e_d=None, e_p_d=None, save_as: str = None):
    data = [
        (e_zz, e_d, 'Total $\epsilon^{tot}_d$', 'g', None),
        (e_zz, e_e_d, 'Elastic $\epsilon^e_d$', 'b', None),
        (e_zz, e_p_d, 'Plastic $\epsilon^p_d$', 'r', None),
    ]
    return create_plot(data, 'Axial Strain $\epsilon_{zz}$', 'Deviatoric Strain $\epsilon_d$', 'Deviatoric Strain vs Axial Strain', save_as)

def plot_volumetric_strain_vs_deviatoric_strain(e_v, e_d, e_e_d=None, e_e_v=None, e_p_d=None, e_p_v=None, save_as: str = None):
    data = [
        (e_d * 100, e_v * 100, 'Total Strain', 'g', None),
        (e_e_d * 100, e_e_v * 100, 'Elastic Strain', 'b', None),
        (e_p_d * 100, e_p_v * 100, 'Plastic Strain', 'r', None),
    ]
    return create_plot(data, '$sign(\epsilon_{zz}) \cdot$ Deviatoric Strain $\epsilon_d$ [%]', 'Volumetric Strain $\epsilon_v$ [%]', '', save_as)

def plot_e_p_eq_vs_e_zz(e_p_eq, e_zz, e_p_eq_calc=None, ok=False, save_as: str = None):
    data = [
        (e_zz, e_p_eq, 'Equivalent Plastic Strain', 'g', None),
        # (e_zz, e_p_eq_calc if ok and e_p_eq_calc is not None else None, 'Calculated Plastic Strain', 'b', None),
    ]
    return create_plot(data, 'Axial Strain $\epsilon_{zz}$', 'Equivalent Plastic Strain $e^p_{eq}$', 'Equivalent Plastic Strain vs Axial Strain', save_as)

def plot_e_xx_vs_e_zz(e_xx, e_zz, save_as: str = None):
    data = [
        (e_zz, e_xx, 'Lateral Strain $\epsilon_{xx}=\epsilon_{yy}$', 'g', None),
    ]
    return create_plot(data, 'Axial Strain $\epsilon_{zz}$', 'Lateral Strain $\epsilon_{xx}=\epsilon_{yy}$', '$\epsilon_{xx}=\epsilon_{yy}$ - Axial Strain', save_as=save_as)

def plot_sig_vs_time(t, sig_1, sig_2, sig_3, save_as: str = None):
    data = [
        (t, sig_1, 'Principal Stress $\sigma_1$', 'r', None),
        (t, sig_2, 'Principal Stress $\sigma_2$', 'g', None),
        (t, sig_3, 'Principal Stress $\sigma_3$', 'b', None),
    ]
    return create_plot(data, 'Time $t$', 'Principal Stresses', 'Principal Stresses vs Time', save_as)


# %%
# image_files = []
PLOT_DIR = Path(f"/mofem_install/jupyter/thomas/mfront_interface/mtest_plots/{model}")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print(f"exx: {e_xx[-1]}")
print(f"eyy: {e_yy[-1]}")
print(f"ezz: {e_zz[-1]}")
print(f"sig1: {sig_1[-1]}")
print(f"sig2: {sig_2[-1]}")
print(f"sig3: {sig_3[-1]}")
print(f"p: {p[-1]}")
print(f"J2: {J_2[-1]}")
print(f"J: {J[-1]}")
print(f"sig_eq: {sig_eq[-1]}")
# plot_stress_field(sig_1, sig_2, sig_3, tau_oct, tau_oct_0,elev=np.degrees(np.arccos(np.sqrt(2/3))),azim=45, title="Deviatoric Plane", save_as=f"{PLOT_DIR}/100_stress_field.png")
# plot_stress_field(sig_1, sig_2, sig_3, tau_oct, tau_oct_0,elev=0,azim=0, title="Biaxial Plane", save_as=f"{PLOT_DIR}/101_stress_field.png")

plot_x_ys(e_zz, [sig_eq], labels=[""], x_label='Axial Strain $\epsilon_{zz}$', y_label='Equivalent Stress $\sigma_{eq}$', title='Equivalent Stress $\sigma_{eq}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{PLOT_DIR}/200_sigeq_ezz.png")
plot_x_ys(ltime, [sig_eq], labels=[""], x_label='Time $t$', y_label='Equivalent Stress $\sigma_{eq}$', title='Equivalent Stress $\sigma_{eq}$ vs Time', save_as = f"{PLOT_DIR}/211_sigeq_t.png")
plot_x_ys(p, [sig_eq], labels=[""], x_label='Time $t$', y_label='Equivalent Stress $\sigma_{eq}$', title='Equivalent Stress $\sigma_{eq}$ vs Time', save_as = f"{PLOT_DIR}/212_sigeq_p.png")
plot_x_ys(e_zz, [sig_eq/p], labels=[""], x_label='Axial Strain $\epsilon_{zz}$', y_label='Stress Ratio $q/p$', title='Equivalent Stress $\sigma_{eq}$ vs Time', save_as = f"{PLOT_DIR}/222_qp_e_xx.png")
label = f"""von Mises (Implicit DSL) - MTest
$E = {E}$
$\\nu = {nu}$
$\\sigma_y$ = {sig_y}
H = {H}"""

plot_x_ys(e_zz * 100, [sig_zz], labels=[label], x_label='Axial Strain $\epsilon_{zz}$ [%]', y_label='Uniaxial Stress $\sigma_{zz}$', title='', save_as = f"{PLOT_DIR}/201_sigzz_ezz.png", handle="off")
plot_x_ys(e_yy, [sig_zz], labels=[label], x_label='Axial Strain $\epsilon_{yy}$', y_label='Stress zz $\sigma_{zz}$', title='Stress $\sigma_{zz}$ vs Axial Strain $\epsilon_{yy}$', save_as = f"{PLOT_DIR}/202_sigzz_eyy.png")
plot_x_ys(e_zz, [sig_1-sig_3], labels=[label], x_label='Axial Strain $\epsilon_{zz}$', y_label='tau', title='Stress $\sigma_{zz}$ vs tau', save_as = f"{PLOT_DIR}/211_sigzz_t1.png")
plot_x_ys(e_zz, [sig_yy], labels=["test1"], x_label='Axial Strain $\epsilon_{zz}$', y_label='Stress yy $\sigma_{yy}$', title='Stress $\sigma_{yy}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{PLOT_DIR}/202_sigyy_ezz.png")
plot_x_ys(e_zz, [sig_xx], labels=["test1"], x_label='Axial Strain $\epsilon_{zz}$', y_label='Stress xx $\sigma_{xx}$', title='Stress $\sigma_{xx}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{PLOT_DIR}/203_sigxx_ezz.png")


plot_x_ys(e_zz, [e_xx, e_yy], labels=["e_xx", "_yy"], x_label='Axial Strain $\epsilon_{zz}$', y_label='Transverse Strain $\epsilon_{xx}=\epsilon_{yy}$', title='Transverse Strain $\epsilon_{xx}=\epsilon_{yy}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{PLOT_DIR}/301_exxyy_ezz.png")
plot_volumetric_strain_vs_deviatoric_strain(e_v=e_v, e_d=np.sign(e_zz) * e_d, e_e_v=e_e_v, e_e_d=np.sign(e_zz) * e_e_d, e_p_v=e_p_v, e_p_d=np.sign(e_zz) * e_p_d,save_as=f"{PLOT_DIR}/304_ev_ed.png")
# plot_volumetric_strain_vs_axial_strain(e_zz=e_zz, e_v=e_v, e_e_v=e_e_v, e_p_v=e_p_v,save_as=f"{PLOT_DIR}/302_ev_ezz.png")
# plot_deviatoric_strain_vs_axial_strain(e_zz=e_zz, e_d=e_d, e_e_d=e_e_d, e_p_d=e_p_d,save_as=f"{PLOT_DIR}/303_ed_ezz.png")
# plot_volumetric_strain_vs_deviatoric_strain(e_v=e_v, e_d=e_d, e_e_v=e_e_v, e_e_d=e_e_d, e_p_v=e_p_v, e_p_d=e_p_d,save_as=f"{PLOT_DIR}/304_ev_ed.png")
# image_files.append()
# image_files.append(plot_stress_field(sig_1, sig_2, sig_3, tau_oct, tau_oct_0,elev=45,azim=20,save_as="1_stress_field_alt2.png"))
# image_files.append(plot_stress_field(sig_1, sig_2, sig_3, tau_oct, tau_oct_0,elev=90,azim=0,save_as="1_stress_field_alt3.png"))
# image_files.append(plot_J_vs_axial_strain(J, e_zz,save_as="2_J_ezz"))
# image_files.append(plot_tau_vs_e_zz(tau_oct, e_zz,save_as="3_tau_ezz"))
# image_files.append(plot_sig_eq_vs_e_zz(sig_eq, e_zz,save_as="4_sigeq_ezz"))
# image_files.append(plot_sig_eq_vs_e_zz(sig_zz, e_zz,save_as="4_sigeq_ezz"))
# image_files.append(plot_2d_with_animation(e_zz, sig_eq, 'Axial Strain $\epsilon_{zz}$', 'Equivalent Stress $\sigma_{eq}$', '$\sigma_{eq}$ - Axial Strain $\epsilon_{zz}$', plastic_cutoff= (sig_y + H * e_p_eq), save_as="41_sigeq_ezz"))

# image_files.append(plot_e_p_eq_vs_e_zz(e_p_eq, e_zz,save_as="8_epeq_ezz.png"))
# image_files.append(plot_e_xx_vs_e_zz(e_xx, e_zz,save_as="9_exx_ezz.png"))
# image_files.append(plot_sig_vs_time(ltime, sig_1, sig_2, sig_3, save_as="10_sig_t.png"))

# %%
from IPython.display import HTML, IFrame, display

# c = 1
a = 1e-9
# phi = 27

x_1 = sig_1
y_1 = sig_2
z_1 = sig_3


# Convert lists to JavaScript arrays
x_1_js = ', '.join(map(str, x_1))
y_1_js = ', '.join(map(str, y_1))
z_1_js = ', '.join(map(str, z_1))

# Combine HTML and JavaScript to create interactive content within the notebook
interacative_html_js = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Desmos Calculator Debug</title>
    <script src="https://www.desmos.com/api/v1.11/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
</head>
<body>
<div id="calculator" style="width: 1200px; height: 600px;"></div>
<script>
    var elt = document.getElementById('calculator');
    var calculator = Desmos.Calculator3D(elt);
    calculator.setExpression({{id:'exp1', latex: 'I = x + y + z'}});
    calculator.setExpression({{id:'exp2', latex: 'p = I / 3'}});
    calculator.setExpression({{id:'exp3', latex: 'J_2= \\\\frac{{1}}{{6}} \\\\cdot ((x-y)^2 + (y-z)^2 + (z-x)^2) '}});
    calculator.setExpression({{id:'exp4', latex: 'q = \\\\sqrt{{3 \\\\cdot J_2}}'}});
    calculator.setExpression({{id:'exp5', latex: 's_{{{0}}}={sig_y}'}});
    calculator.setExpression({{id:'exp8', latex: 'H = {H}'}});

    calculator.setExpression({{id:'exp9', 
    latex: '0 = q - s_{0}',
    color: Desmos.Colors.RED,
    }});

    calculator.setExpression({{
        type: 'table',
        columns: [
            {{
                latex: 'x_1',
                values: [{x_1_js}]
            }},
            {{
                latex: 'y_1',
                values: [{y_1_js}],
            }},
            {{
                latex: 'z_1',
                values: [{z_1_js}],
            }},
        ]
    }});

    calculator.setExpression({{id:'exp11', 
    latex: '(x_{{1}},y_{{1}},z_{{1}})',
    color: Desmos.Colors.BLUE,
    }});
    
    
    function downloadScreenshot() {{
        var screenshot = calculator.screenshot();
        var link = document.createElement('a');
        link.href = screenshot;
        link.download = 'screenshot.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }}
    
</script>
<h2>Interactive Content</h2>
<button onclick="downloadScreenshot()">Click me to download screenshot!</button>
</body>
"""
Html_file= open(f"{PLOT_DIR}/Desmos3D.html","w")
Html_file.write(interacative_html_js)
Html_file.close()


