from setuptools import setup, find_packages

# Setting Up

setup(
      
      name = "anaplan",
      version = "1.5.1",
      author = "Hasan Can Beydili",
      author_email = "tchasancan@gmail.com",
      description= "learner function have new parameters: 'show_current_activations' and learner functions 'strategy' parameter have new options: 'f1', 'precision', 'recall', 'all'. For more details read learner functions doc string.: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf.",
      packages=find_packages(),
      package_data={
        'anaplan': ['*.mp4']
      },
      include_package_data=True,
      keywords = ["model evaluation", "classifcation", 'potentiation learning artficial neural networks'],

      
      )