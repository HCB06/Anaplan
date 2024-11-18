# anaplan/plan/__init__.py

# Bu dosya, plan modülünün ana giriş noktasıdır.

import subprocess
import pkg_resources

print("Auto checking and installation dependencies for Anaplan")

package_names = [
    'scipy==1.13.1',
    'tqdm==4.66.4',
    'sounddevice==0.5.0',
    'soundfile==0.12.1',
    'seaborn==0.13.2',
    'pandas==2.2.2',
    'moviepy==1.0.3',
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

print(f"Anaplan is ready to use with {err} errors")

__version__ = "1.6" 
__update__ = "*'encode_one_hot' function has new parameter: 'summary'.\n* 'draw_model_architecture' function updated.\n*. For more details: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf."

def print_version(__version__):
    print(f"Anaplan Version {__version__}" + '\n')

def print_update_notes(__update__):
    print(f"Update Notes:\n{__update__}")

print_version(__version__)
print_update_notes(__update__)

from .plan import *