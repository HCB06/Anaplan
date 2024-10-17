# anaplan/plan/__init__.py

# Bu dosya, plan modülünün ana giriş noktasıdır.

__version__ = "1.5.1" 
__update__ = "learner function have new parameters: 'show_current_activations' and learner functions 'strategy' parameter have new options: 'f1', 'precision', 'recall', 'all'. For more details read learner functions doc string.: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf."

def print_version(__version__):
    print(f"Anaplan Version {__version__}" + '\n')

def print_update_notes(__update__):
    print(__update__)

print_version(__version__)
print_update_notes(__update__)

from .plan import *