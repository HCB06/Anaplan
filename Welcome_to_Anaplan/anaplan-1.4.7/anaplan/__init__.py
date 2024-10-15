# anaplan/plan/__init__.py

# Bu dosya, plan modülünün ana giriş noktasıdır.

__version__ = "1.4.7" 

def print_version():
    print(f"Anaplan Version {__version__}" + '\n')

from .plan import *