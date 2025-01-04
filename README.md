# PyerualJetwork [![Socket Badge](https://socket.dev/api/badge/pypi/package/anaplan/2.5.3?artifact_id=tar-gz)](https://socket.dev/pypi/package/anaplan/overview/2.5.0/tar-gz) [![CodeFactor](https://www.codefactor.io/repository/github/hcb06/PyerualJetwork/badge)](https://www.codefactor.io/repository/github/hcb06/PyerualJetwork) [![PyPI Downloads](https://static.pepy.tech/badge/PyerualJetwork)](https://pepy.tech/projects/PyerualJetwork) + [![PyPI Downloads](https://static.pepy.tech/badge/anaplan)](https://pepy.tech/projects/anaplan) [![PyPI Downloads](https://static.pepy.tech/badge/anaplan/month)](https://pepy.tech/projects/anaplan) [![PyPI Downloads](https://static.pepy.tech/badge/anaplan/week)](https://pepy.tech/projects/anaplan) [![PyPI version](https://img.shields.io/pypi/v/anaplan.svg)](https://pypi.org/project/anaplan/) [![PyPI version](https://img.shields.io/pypi/v/pyerualjetwork.svg)](https://pypi.org/project/pyerualjetwork/)

Note: anaplan old name of pyerualjetwork.


The first version of PyerualJetwork was released on May 22, 2024. Temporary end of support date Aug 22, 2024. (Jan 4, 2025 > SUPPORT RECONTUNIUNG.)

The first version of Anaplan was released on Aug 30, 2024. End of support date Jan 4, 2025.


IMPORTANT CHANGE!

I recently updated the name of the library I published as "pyerualjetwork" to "anaplan." However, due to copyright issues and branding concerns, I have decided to revert to the original name, "pyerualjetwork." From now on, you can access the library by installing version pyerualjetwork(3.3.4) using the command pip install pyerualjetwork, replacing the current "anaplan(2.5.3)" modules.

https://libraries.io/pypi/pyerualjetwork (re-continuing support)

https://libraries.io/pypi/anaplan (end of support)

      pip install pyerualjetwork
      
      from pyerualjetwork import plan
      from pyerualjetwork import planeat
      from pyerualjetwork import data_operations
      from pyerualjetwork import model_operations

      Optimized for Visual Studio Code(Note: Improved for other ides in 1.5.8>.)
      
      requires=[
        'setuptools==75.6.0'
 	    'scipy==1.13.1',
	    'tqdm==4.66.4',
	    'seaborn==0.13.2',
	    'pandas==2.2.2',
	    'networkx==3.3',
	    'numpy==1.26.4',
	    'matplotlib==3.9.0',
	    'colorama==0.4.6'
        ]

     matplotlib, seaborn, networkx (optional).
     PyerualJetwork checks and install all dependencies (with optional ones) for every runing.
     If your version is higher or lower, PyerualJetwork automaticly delete other versions and installs this versions.
          
##############################

ABOUT PYERUALJETWORK:

PyerualJetwork is a machine learning library written in Python for professionals, incorporating advanced, unique, new, and modern techniques. Its most important component is the PLAN (Potentiation Learning Artificial Neural Network) https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4862342. (THIS ARTICLE IS FIRST VERSION OF PLAN.) MODERN VERSION OF PLAN: https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PLAN/PLAN.pdf
Both the PLAN algorithm and the PyerualJetwork library were created by Author, and all rights are reserved by Author.
PyerualJetwork is free to use for commercial business and individual users. PyerualJetwork is written in fully functional programming with non-oop elements. PyerualJetwork consists of many functions that complement each other, which facilitates the learning process and debugging during use.
As of 12/21/2024, the library includes PLAN and PLANEAT module, but other machine learning modules are expected to be added in the future.
<br><br>

PyerualJetwork includes Plan Vision, NLPlan, PLANEAT and at the between of both, Deep Plan:<br>

![PyerualJetwork](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PyerualJetwork.jpg)<br><br><br>

PLAN VISION:<br>

![PLAN VISION](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PlanVision.jpg)

You can create artificial intelligence models that perform computer vision tasks using the plan module:<br>

![AUTONOMOUS](https://github.com/HCB06/PyerualJetwork/blob/main/Media/autonomous.gif)<br><br><br>
![XRAY](https://github.com/HCB06/PyerualJetwork/blob/main/Media/chest_xray.png)<br><br><br>
![GENDER](https://github.com/HCB06/PyerualJetwork/blob/main/Media/gender_classification.png)<br><br><br>

NLPlan:<br>

![NLPLAN](https://github.com/HCB06/PyerualJetwork/blob/main/Media/NLPlan.jpg)<br>

You can create artificial intelligence models that perform natural language processing tasks using the plan module:

![PLAN VISION](https://github.com/HCB06/PyerualJetwork/blob/main/Media/NLP.gif)

PLANEAT:<br>

You can create artificial intelligence models that perform reinforcement learning tasks and genetic optimization tasks using the planeat module:

![PLANEAT](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PLANEAT_1.gif)<br>

![PLANEAT](https://github.com/HCB06/PyerualJetwork/blob/main/Media/PLANEAT_2.gif)<br>


HOW DO I IMPORT IT TO MY PROJECT?

Anaconda users can access the 'Anaconda Prompt' terminal from the Start menu and add the necessary library modules to the Python module search queue by typing "pip install PyerualJetwork" and pressing enter. If you are not using Anaconda, you can simply open the 'cmd' Windows command terminal from the Start menu and type "pip install pyerualjetwork". (Visual Studio Code reccomended) After installation, it's important to periodically open the terminal of the environment you are using and stay up to date by using the command "pip install pyerualjetwork --upgrade".
•
After installing the module using "pip" you can now call the library module in your project environment. Use: “from pyerualjetwork import plan”. Now, you can call the necessary functions from the plan module.

The PLAN algorithm will not be explained in this document. This document focuses on how professionals can integrate and use PyerualJetwork in their systems. However, briefly, the PLAN algorithm can be described as a classification algorithm. PLAN algorithm achieves this task with an incredibly energy-efficient, fast, and hyperparameter-free user-friendly approach. For more detailed information, you can check out ![PyerualJetwork USER MANUEL](https://github.com/HCB06/PyerualJetwork/blob/main/Welcome_to_PyerualJetwork/PYERUALJETWORK_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf) file.
