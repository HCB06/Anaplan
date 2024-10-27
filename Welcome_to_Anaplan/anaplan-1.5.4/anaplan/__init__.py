# anaplan/plan/__init__.py

# Bu dosya, plan modülünün ana giriş noktasıdır.

__version__ = "1.5.4" 
__update__ = "*'big_data_mode' parameter of learner function changed to 'auto_normalization'. Default=True.\n*'batch_size' parameter added for learner function.\nFor more details read learner functions doc string.: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf."

def print_version(__version__):
    print(f"Anaplan Version {__version__}" + '\n')

def print_update_notes(__update__):
    print(__update__)

print_version(__version__)
print_update_notes(__update__)

from .plan import *