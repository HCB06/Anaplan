from .activation_functions import all_activations

def activation_potentiation():

    activations_list = all_activations()

    print('All available activations: ',  activations_list, "\n\nYOU CAN COMBINE EVERY ACTIVATION. EXAMPLE: ['linear', 'tanh'] or ['waveakt', 'linear', 'sine'].")

    return activations_list

def plan():

    print('PLAN document and examples: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_PLAN')

def planeat():

    print('PLANEAT examples: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_Anaplan/ExampleCodes/PLANEAT')


def anaplan():

    print('Anaplan document and examples: https://github.com/HCB06/Anaplan/tree/main/Welcome_to_Anaplan')