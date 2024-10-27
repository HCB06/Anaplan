from setuptools import setup, find_packages

# Setting Up

setup(
      
      name = "anaplan",
      version = "1.5.6",
      author = "Hasan Can Beydili",
      author_email = "tchasancan@gmail.com",
      description= "*'big_data_mode' parameter of learner function changed to 'auto_normalization'. Default=True.\n*'batch_size' parameter added for learner function.\nFor more details read learner functions doc string.: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf.",
      packages=find_packages(),
      package_data={
        'anaplan': ['*.mp4']
      },
      include_package_data=True,
      keywords = ["model evaluation", "classifcation", 'potentiation learning artficial neural networks'],

      
      )