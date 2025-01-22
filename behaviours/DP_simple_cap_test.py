import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Importing the data directly from the file path provided
file_path = "/mofem_install/jupyter/thomas/mfront_interface/DP_simple_Cap.res"
c = 0.3
phi = 0.26
a = 0
PLOT_DIR = "/mofem_install/jupyter/thomas/mfront_interface/mtest_plots/DruckerPragerCap"
os.makedirs(PLOT_DIR, exist_ok=True)
# Read the data from the specified file path
data = pd.read_csv(file_path, sep='\s+', skiprows=28, header=None)

# Assign column names to match the provided description
data.columns = [
    "time", "EXX", "EYY", "EZZ", "EXY", "EXZ", "EYZ", "SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ",
    "ElasticStrain_1", "ElasticStrain_2", "ElasticStrain_3", "ElasticStrain_4", "ElasticStrain_5", "ElasticStrain_6",
    "EquivalentPlasticStrain[0]", "EquivalentPlasticStrain[1]", "StoredEnergy", "DissipatedEnergy"
]

# Extract stress components
sig_xx = data["SXX"].values
sig_yy = data["SYY"].values
sig_zz = data["SZZ"].values
sig_xy = data["SXY"].values
sig_xz = data["SXZ"].values
sig_yz = data["SYZ"].values
e_xx = data["EXX"].values
e_yy = data["EYY"].values
e_zz = data["EZZ"].values
e_xy = data["EXY"].values
e_xz = data["EXZ"].values
e_yz = data["EYZ"].values
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

e_v, e_d = calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)

# Define the function to calculate principal stresses
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

# Calculate the principal stresses
sig_1, sig_2, sig_3 = calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz)

# Display the results
principal_stresses_df = pd.DataFrame({
    "time": data["time"],
    "sig_1": sig_1,
    "sig_2": sig_2,
    "sig_3": sig_3
})

def create_plot(data, x_label, y_label, title, save_as):
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
            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
    
    # Add axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    ax.grid(True)

    if save_as:
        plt.savefig(save_as)
        return save_as

def plot_volumetric_strain_vs_deviatoric_strain(e_v, e_d, e_e_d=None, e_e_v=None, e_p_d=None, e_p_v=None, save_as: str = None):
    data = [
        (e_d, e_v, 'Total Strain', 'g', None),
    ]
    return create_plot(data, 'Deviatoric Strain $\epsilon_d$', 'Volumetric Strain $\epsilon_v$', 'Volumetric Strain vs Deviatoric Strain', save_as)

plot_volumetric_strain_vs_deviatoric_strain(e_v=e_v, e_d=e_d, save_as=f"{PLOT_DIR}/304_ev_ed.png")


from IPython.display import HTML


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
    calculator.setExpression({{id:'exp5', latex: 'p_{{hi}} = {phi}'}});
    calculator.setExpression({{id:'exp6', latex: 'M_{{JP}} = \\\\frac{{2\\\\sqrt{{3}}\\\\sin p_{{hi}}}}{{3-\\\\sin p_{{hi}}}}'}});
    calculator.setExpression({{id:'exp7', latex: 'a = {a}'}});
    calculator.setExpression({{id:'exp8', latex: 'c = {c}'}});

    calculator.setExpression({{id:'exp9', 
    latex: '0 = + M_{{JP}} p + \\\\sqrt{{a^{{2}} M_{{JP}}^{{2}} + \\\\frac{{q}}{{\\\\sqrt{{3}}}}^{{2}}}} - M_{{JP}} \\\\cdot \\\\frac{{c}}{{\\\\tan p_{{hi}}}}',
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
Html_file= open(f"{PLOT_DIR}/Desmos3D_raw.html","w")
Html_file.write(interacative_html_js)
Html_file.close()

# e_p_xx = data["PlasticStrain_1"].values
# e_p_yy = data["PlasticStrain_2"].values
# e_p_zz = data["PlasticStrain_3"].values
# e_p_xy = data["PlasticStrain_4"].values
# e_p_xz = data["PlasticStrain_5"].values
# e_p_yz = data["PlasticStrain_6"].values

# e_p_1, e_p_2, e_p_3 = calculate_principal_stresses(e_p_xx, e_p_yy, e_p_zz, e_p_xy, e_p_xz, e_p_yz)

# # Convert lists to JavaScript arrays
# e_p_1 = ', '.join(map(str, e_p_1))
# e_p_2 = ', '.join(map(str, e_p_2))
# e_p_3 = ', '.join(map(str, e_p_3))


# interacative_html_js = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Desmos Calculator Debug</title>
#     <script src="https://www.desmos.com/api/v1.11/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
# </head>
# <body>
# <div id="calculator" style="width: 1200px; height: 600px;"></div>
# <script>
#     var elt = document.getElementById('calculator');
#     var calculator = Desmos.Calculator3D(elt);
#     calculator.setExpression({{id:'exp1', latex: 'I = x + y + z'}});
#     calculator.setExpression({{id:'exp2', latex: 'p = I / 3'}});
#     calculator.setExpression({{id:'exp3', latex: 'J_2= \\\\frac{{1}}{{6}} \\\\cdot ((x-y)^2 + (y-z)^2 + (z-x)^2) '}});
#     calculator.setExpression({{id:'exp4', latex: 'q = \\\\sqrt{{3 \\\\cdot J_2}}'}});
#     calculator.setExpression({{id:'exp5', latex: 'p_{{hi}} = {phi}'}});
#     calculator.setExpression({{id:'exp6', latex: 'M_{{JP}} = \\\\frac{{2\\\\sqrt{{3}}\\\\sin p_{{hi}}}}{{3-\\\\sin p_{{hi}}}}'}});
#     calculator.setExpression({{id:'exp7', latex: 'a = {a}'}});
#     calculator.setExpression({{id:'exp8', latex: 'c = {c}'}});

#     calculator.setExpression({{id:'exp9', 
#     latex: '0 = + M_{{JP}} p + \\\\sqrt{{a^{{2}} M_{{JP}}^{{2}} + \\\\frac{{q}}{{\\\\sqrt{{3}}}}^{{2}}}} - M_{{JP}} \\\\cdot \\\\frac{{c}}{{\\\\tan p_{{hi}}}}',
#     color: Desmos.Colors.RED,
#     }});

#     calculator.setExpression({{
#         type: 'table',
#         columns: [
#             {{
#                 latex: 'x_1',
#                 values: [{e_p_1}]
#             }},
#             {{
#                 latex: 'y_1',
#                 values: [{e_p_2}],
#             }},
#             {{
#                 latex: 'z_1',
#                 values: [{e_p_3}],
#             }},
#         ]
#     }});

#     calculator.setExpression({{id:'exp11', 
#     latex: '(x_{{1}},y_{{1}},z_{{1}})',
#     color: Desmos.Colors.BLUE,
#     }});
    
    
#     function downloadScreenshot() {{
#         var screenshot = calculator.screenshot();
#         var link = document.createElement('a');
#         link.href = screenshot;
#         link.download = 'screenshot.png';
#         document.body.appendChild(link);
#         link.click();
#         document.body.removeChild(link);
#     }}
    
# </script>
# <h2>Interactive Content</h2>
# <button onclick="downloadScreenshot()">Click me to download screenshot!</button>
# </body>
# """
# Html_file= open(f"{PLOT_DIR}/Desmos3D_raw_plastic_strain.html","w")
# Html_file.write(interacative_html_js)
# Html_file.close()
