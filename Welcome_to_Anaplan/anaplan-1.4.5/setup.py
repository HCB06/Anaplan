from setuptools import setup, find_packages

# Setting Up

setup(
      
      name = "anaplan",
      version = "1.4.5",
      author = "Hasan Can Beydili",
      author_email = "tchasancan@gmail.com",
      description= "learner and fit function have new parameter: 'neurons_history'. It shows neurons history. fit function and learner function's issues solved. : https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf.",
      packages=find_packages(),
      package_data={
        'anaplan': ['*.mp4']
      },
      include_package_data=True,
      keywords = ["model evaluation", "classifcation", 'potentiation learning artficial neural networks'],

      
      )