from setuptools import setup, find_packages

# Setting Up

setup(
      
      name = "anaplan",
      version = "1.1.7",
      author = "Hasan Can Beydili",
      author_email = "tchasancan@gmail.com",
      description= "Voice information added(expr.)",
      packages=find_packages(),
      package_data={
        'anaplan': ['*.mp4']
      },
      include_package_data=True,
      keywords = ["model evaluation", "classifcation", 'potentiation learning artficial neural networks'],

      
      )