
import subprocess
subprocess.check_call(["pip", "install", 'setuptools==75.6.0'])
import pkg_resources
from datetime import datetime

print("Auto checking and installation dependencies for PyerualJetwork")

package_names = [
    'scipy==1.13.1',
    'tqdm==4.66.4',
    'seaborn==0.13.2',
    'pandas==2.2.2',
    'networkx==3.3',
    'numpy==1.26.4',
    'matplotlib==3.9.0',
    'colorama==0.4.6'
]

installed_packages = pkg_resources.working_set
installed = {pkg.key: pkg.version for pkg in installed_packages}
err = 0

for package_name in package_names:
    package_name_only, required_version = package_name.split('==')
    
    if package_name_only not in installed:

        try:
            print(f"{package_name} Installing...")
            subprocess.check_call(["pip", "install", package_name])
        except Exception as e:
            err += 1
            print(f"Error installing {package_name} library, installation continues: {e}")
    else:

        installed_version = installed[package_name_only]
        if installed_version != required_version:
            print(f"Updating {package_name_only} from version {installed_version} to {required_version}...")
            try:
                subprocess.check_call(["pip", "install", package_name])
            except Exception as e:
                err += 1
                print(f"Error updating {package_name} library, installation continues: {e}")
        else:
            print(f"{package_name} ready.")

print(f"PyerualJetwork is ready to use with {err} errors")

__version__ = "4.0.9"
__update__ = "* Note: CUDA modules need cupy. Enter this command in your terminal: 'pip install cupy-cuda12x' or your cuda version.\n* Changes: https://github.com/HCB06/PyerualJetwork/blob/main/CHANGES\n* PyerualJetwork document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf\n* YouTube tutorials: https://www.youtube.com/@HasanCanBeydili"

def print_version(__version__):
    print(f"PyerualJetwork Version {__version__}" + '\n')

def print_update_notes(__update__):
    print(f"Update Notes:\n{__update__}")

print_version(__version__)
print_update_notes(__update__)

from .plan import *
from .planeat import *
from .activation_functions import *
from .data_operations import *
from .loss_functions import *
from .metrics import *
from .model_operations import *
from .ui import *
from .visualizations import *
from .help import *