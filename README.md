# PyerualJetwork 

https://libraries.io/pypi/pyerualjetwork

      requires=[
          'numpy',
          'scipy',
            'time',
            'math',
            'colorama',
            'typing'
            ],
      
       extras_require={
          'visualization': ['matplotlib','seaborn']
          
##############################

2.0.8 New features: BINARY INJECTION (OLD) MODULE, NOW ADDED NEW DIRECT FEATURE INJECTION MODULE. AND 'standard_scaler' func. Important Note: If there are any data smaller than 0 among the input data of the entry model, "import plan_bi"; otherwise, "import plan_di". 

PYERUAL JETWORK 2.0 USER MANUAL

Author: Hasan Can Beydili

ABOUT PYERUAL JETWORK:

Pyerual Jetwork is a machine learning library written in Python for professionals, incorporating advanced, unique, new, and modern techniques. Its most important component is the PLAN (Pruning Learning Artificial Neural Network).

Both the PLAN algorithm and the Pyerual Jetwork library were created by Hasan Can Beydili, and all rights are reserved by Hasan Can Beydili.

As of 05/24/2024, the library includes only the PLAN module, but other machine learning modules are expected to be added in the future.

The PLAN algorithm will not be explained in this document. This document focuses on how professionals can integrate and use Pyerual Jetwork in their systems. However, briefly, the PLAN algorithm can be described as a classification algorithm. For more detailed information, you can check out 'Welcome to PLAN' folder.

The functions of the Pyerual Jetwork modules, uses snake_case written style.

The PLAN module consists of 22 functions. Of these, 13 are main functions (which the user will interact with frequently), and the remaining 9 are auxiliary functions.

HOW DO I IMPORT IT TO MY PROJECT?

Anaconda users can access the 'Anaconda Prompt' terminal from the Start menu and add the necessary library modules to the Python module search queue by typing "pip install pyerualjetwork" and pressing enter. If you are not using Anaconda, you can simply open the 'cmd' Windows command terminal from the Start menu and type "pip install pyerualjetwork". After installation, it's important to periodically open the terminal of the environment you are using and stay up to date by using the command "pip install pyerualjetwork --upgrade". As of 06/04/2024, the most current version is "2.0.5". This version is the successor to version "1.0" of the library. Previous versions were for testing purposes.
After installing the modules using "pip" you can now call the library modules in your project environment. To do this, simply write "import plan" in the project where you're writing your code. Alternatively, you can use "import plan as p". Now, you can call the necessary functions from the plan module.
MAIN FUNCTIONS:
1. fit (3Args)
2. evaluate (5Args)
3. save_model (9Args)
4. load_model (2Args)
5. predict_model_ssd (3Args)
6. predict_model_ram (3Args)
7. auto_balancer (3Args)
8. synthetic_augmentation (3Args)
9. get_weights ()
10. get_df ()
11. get_preds ()
12. get_acc ()
13. get_pot ()
-----

Other details in the 'Welcome to Pyerual Jetwork' folder
