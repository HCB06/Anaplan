# Anaplan [![Socket Badge](https://socket.dev/api/badge/pypi/package/anaplan/2.4.9?artifact_id=tar-gz)](https://socket.dev/pypi/package/anaplan/overview/2.4.9/tar-gz) [![PyPI Downloads](https://static.pepy.tech/badge/anaplan)](https://pepy.tech/projects/anaplan) + [![PyPI Downloads](https://static.pepy.tech/badge/pyerualjetwork)](https://pepy.tech/projects/pyerualjetwork) [![PyPI Downloads](https://static.pepy.tech/badge/anaplan/month)](https://pepy.tech/projects/anaplan) [![PyPI Downloads](https://static.pepy.tech/badge/anaplan/week)](https://pepy.tech/projects/anaplan)

Note: pyerualjetwork old name of anaplan.

https://libraries.io/pypi/anaplan

      pip install anaplan
      
      from anaplan import plan
      from anaplan import planeat
      from anaplan import data_operations
      from anaplan import model_operations

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
     Anaplan checks and install all dependencies (with optional ones) for every runing.
     If your version is higher or lower, Anaplan automaticly delete other versions and installs this versions.
          
##############################

ABOUT ANAPLAN:

Anaplan is a machine learning library written in Python for professionals, incorporating advanced, unique, new, and modern techniques. Its most important component is the PLAN (Potentiation Learning Artificial Neural Network) https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4862342. (THIS ARTICLE IS FIRST VERSION OF PLAN.) MODERN VERSION OF PLAN: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_PLAN/PLAN.pdf
Both the PLAN algorithm and the Anaplan library were created by Author, and all rights are reserved by Author.
Anaplan is free to use for commercial business and individual users. Anaplan is written in fully functional programming with non-oop elements. Anaplan consists of many functions that complement each other, which facilitates the learning process and debugging during use.
As of 12/21/2024, the library includes PLAN and PLANEAT module, but other machine learning modules are expected to be added in the future.
<br><br>

Anaplan includes Plan Vision, NLPlan, PLANEAT and at the between of both, Deep Plan:<br>

![ANAPLAN](https://github.com/HCB06/PyerualJetwork/blob/main/Media/anaplan.jpg)<br><br><br>

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

![PLANEAT](https://github.com/HCB06/Anaplan/blob/main/Media/PLANEAT_1.gif)<br>

![PLANEAT](https://github.com/HCB06/Anaplan/blob/main/Media/PLANEAT_2.gif)<br>


HOW DO I IMPORT IT TO MY PROJECT?

Anaconda users can access the 'Anaconda Prompt' terminal from the Start menu and add the necessary library modules to the Python module search queue by typing "pip install anaplan" and pressing enter. If you are not using Anaconda, you can simply open the 'cmd' Windows command terminal from the Start menu and type "pip install anaplan". (Visual Studio Code reccomended) After installation, it's important to periodically open the terminal of the environment you are using and stay up to date by using the command "pip install anaplan --upgrade". The latest version was “1.0.1” at the time this document was written
•
After installing the module using "pip" you can now call the library module in your project environment. Use: “from anaplan import plan”. Now, you can call the necessary functions from the plan module.

The PLAN algorithm will not be explained in this document. This document focuses on how professionals can integrate and use Anaplan in their systems. However, briefly, the PLAN algorithm can be described as a classification algorithm. PLAN algorithm achieves this task with an incredibly energy-efficient, fast, and hyperparameter-free user-friendly approach. For more detailed information, you can check out ![ANAPLAN USER MANUEL](https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf) file.
