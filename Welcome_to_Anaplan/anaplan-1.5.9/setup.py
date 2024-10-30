from setuptools import setup, find_packages

# Setting Up

setup(
      
      name = "anaplan",
      version = "1.5.9",
      author = "Hasan Can Beydili",
      author_email = "tchasancan@gmail.com",
      description= "* '* 'learner function improved.\n* learner function has new parameter: 'neural_web_history'.\n* save_model function has new parameter: 'show_architecture'.\n * plan module has new functions: 'draw_model_architecture' and 'draw_neural_web'. For more details: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf.",
      packages=find_packages(),
      package_data={
        'anaplan': ['*.mp4']
      },
      include_package_data=True,
      keywords = ["model evaluation", "classifcation", 'potentiation learning artficial neural networks'],

      
      )