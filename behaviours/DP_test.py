import pandas as pd
import numpy as np

# Importing the data directly from the file path provided
file_path = "/mofem_install/jupyter/thomas/mfront_interface/DP_simple.res"

# Read the data from the specified file path
data = pd.read_csv(file_path, sep='\s+', skiprows=28, header=None)

# Assign column names to match the provided description
data.columns = [
    "time", "EXX", "EYY", "EZZ", "EXY", "EXZ", "EYZ", "SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ",
    "ElasticStrain_1", "ElasticStrain_2", "ElasticStrain_3", "ElasticStrain_4", "ElasticStrain_5", "ElasticStrain_6",
    "EquivalentPlasticStrain", "PlasticStrain_1", "PlasticStrain_2", "PlasticStrain_3",
    "PlasticStrain_4", "PlasticStrain_5", "PlasticStrain_6", "StoredEnergy", "DissipatedEnergy"
]

# Extract stress components
sig_xx = data["SXX"].values
sig_yy = data["SYY"].values
sig_zz = data["SZZ"].values
sig_xy = data["SXY"].values
sig_xz = data["SXZ"].values
sig_yz = data["SYZ"].values

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

from IPython.display import HTML

c = 20
phi = 0.26
a = 0.001
PLOT_DIR = "/mofem_install/jupyter/thomas/mfront_interface/mtest_plots/DruckerPragerSimple"
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