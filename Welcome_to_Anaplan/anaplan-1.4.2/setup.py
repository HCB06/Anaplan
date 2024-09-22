from setuptools import setup, find_packages

# Setting Up

setup(
      
      name = "anaplan",
      version = "1.4.2",
      author = "Hasan Can Beydili",
      author_email = "tchasancan@gmail.com",
      description= "learner functions new parameters: 'big_data_mode', 'early_shifting'. And better optimized module codes with incredible speed up.: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf.",
      packages=find_packages(),
      package_data={
        'anaplan': ['*.mp4']
      },
      include_package_data=True,
      keywords = ["model evaluation", "classifcation", 'potentiation learning artficial neural networks'],

      
      )