""" 
MAIN MODULE FOR PLANEAT

ANAPLAN document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydili.06/
@contact: tchasancan@gmail.com
"""

import numpy as np
import random
from tqdm import tqdm

### LIBRARY IMPORTS ###
from .plan import fex
from .data_operations import normalization
from .ui import loading_bars
from .activation_functions import apply_activation, activations_list

def define_genomes(input_shape, output_shape, population_size):
   """
   Initializes a population of genomes, where each genome is represented by a set of weights 
   and an associated activation function. Each genome is created with random weights and activation 
   functions are applied and normalized. (Max abs normalization.)

   Args:
      input_shape (int): The number of input features for the neural network.
      output_shape (int): The number of output features for the neural network.
      population_size (int): The number of genomes (individuals) in the population.

   Returns:
      tuple: A tuple containing:
         - population_weights (numpy.ndarray): A 2D numpy array of shape (population_size, output_shape, input_shape) representing the 
            weight matrices for each genome.
         - population_activations (list): A list of activation functions applied to each genome.
            
   Notes:
      The weights are initialized randomly within the range [-1, 1]. 
      Activation functions are selected randomly from a predefined list `activations_list()`.
      The weights for each genome are then modified by applying the corresponding activation function 
      and normalized using the `normalization()` function. (Max abs normalization.)
   """
   population_weights = []

   activations = [x for x in activations_list() if x not in [4, 5]] # SPIRAL AND CIRCULAR ACTIVATION DISCARDED
   population_activations = []

   for _ in range(population_size):
      population_weights.append(np.random.uniform(-1, 1, (output_shape, input_shape)))

      population_activations.append(activations[int(random.uniform(0, len(activations)-1))])
      population_weights[-1] = apply_activation(population_weights[-1], population_activations[-1])

      population_weights[-1] = normalization(population_weights[-1])

   return np.array(population_weights), population_activations



def learner(weights, activation_potentiations, what_gen, y_reward, show_info=False, strategy='cross_over', policy='normal_selective', mutations=True, bad_genoms_mutation_prob=None, activation_mutate_prob=0.5, save_best_genom=True):
   """
    Applies the learning process of a population of genomes using selection, crossover, mutation, and activation function potentiation.
    The function modifies the population's weights and activation functions based on a specified policy, mutation probabilities, and strategy.

    Args:
        weights (numpy.ndarray): Array of weights for each genoms. (first returned value of define_genomes function)
        activation_potentiations (list): A list of activation functions for each genomes. (second returned value of define_genomes function)
        what_gen (int): The current generation number, used for informational purposes or logging.
        y_reward (numpy.ndarray): A 1D array containing the fitness or reward values of each genome. The array is used to rank the genomes based on their performance. And PLANEAT maximize the reward.
        show_info (bool, optional): If True, prints information about the current generation and the maximum reward obtained. Also shows current configuration. Default is False.
        \n strategy (str, optional): The strategy for combining the best and bad genomes. Options: \n
                                  - 'cross_over': Perform Two-Point Matrix Crossover between the best genomes and replace bad genomes. (Classic NEAT cross over)\n
                                  - 'potentiate': Cumulate the weight of the best genomes and replace bad genomes. (PLAN feature. 'Like Arithmetic Crossover but different.')\n
                                  Default is 'cross_over'.
        \n policy (str, optional): The selection policy that governs how genomes are selected for reproduction. Options: \n
                                 - 'normal_selective': Normal selection based on reward, where a portion of the bad genes are discarded.\n
                                 - 'more_selective': A more selective policy, where fewer bad genes survive.\n
                                 - 'less_selective': A less selective policy, where more bad genes survive.\n
                                 Default is 'normal_selective'.\n
        mutations (bool, optional): If True, mutations are applied to the bad genomes and potentially to the best genomes as well. Default is True.
        bad_genoms_mutation_prob (float, optional): The probability of applying mutation to the bad genomes. Must be in the range [0, 1]. Also effects best genoms mutatation prob. For example 0.7 value for bad genoms then 0.3 value for best genoms. Default is None, which means it is determined by the `policy` argument.
        activation_mutate_prob (float, optional): The probability of applying mutation to the activation functions. Must be in the range [0, 1]. Default is 0.5 (% 50)
        save_best_genom (bool, optional): If True, ensures that the best genome are saved and not mutated or altered during reproduction. Default is True.

    Raises:
        ValueError: 
            - If `policy` is not one of the specified values ('normal_selective', 'more_selective', 'less_selective').
            - If `bad_genoms_mutation_prob` or `activation_mutate_prob` are not in the range [0, 1].
            - If the population size is odd (ensuring an even number of genomes is required for proper selection).

    Returns:
        tuple: A tuple containing:
            - weights (numpy.ndarray): The updated weights for the population after selection, crossover, and mutation. 
                                      The shape is (population_size, output_shape, input_shape).
            - activation_potentiations (list): The updated list of activation functions for the population.

    Notes:
        - **Selection Process**: 
            - The genomes are sorted by their fitness (based on `y_reward`), and then split into "best" and "bad" half. 
            - The best genomes are retained, and the bad genomes are modified based on the selected strategy.
            
        - **Crossover and Potentiation Strategies**:
            - The **'cross_over'** strategy performs crossover, where parts of the best genomes' weights are combined with the other good genomes to create new weight matrices.
            - The **'potentiate'** strategy strengthens the best genomes by potentiating their weights towards the other good genomes.
            
        - **Mutation**:
            - Mutation is applied to both the best and bad genomes, depending on the mutation probability and the `policy`.
            - `bad_genoms_mutation_prob` determines the probability of applying mutations to the bad genomes.
            - If `activation_mutate_prob` is provided, activation function mutations are applied to the genomes based on this probability.
            
        - **Policy Types**:
            - **'normal_selective'**: A typical selection policy where a portion of the bad genomes is discarded. The mutation probability for the bad genomes is set to 0.7 by default.
            - **'more_selective'**: A stricter selection policy where more of the bad genomes are discarded. The mutation probability for the bad genomes is set to 0.85 by default.
            - **'less_selective'**: A more lenient selection policy where fewer bad genomes are discarded. The mutation probability for the bad genomes is set to 0.6 by default.
            
        - **Population Size**: The population size must be an even number to properly split the best and bad genomes. If `y_reward` has an odd length, an error is raised.
        
        - **Logging**: If `show_info=True`, the current generation and the maximum reward from the population are printed for tracking the learning progress.

    Example:
        ```python
        weights, activation_potentiations = learner(weights, activation_potentiations, 1, y_reward, info=True, strategy='cross_over', policy='normal_selective')
        ```

    - The function returns the updated weights and activations after processing based on the chosen strategy, policy, and mutation parameters.
   """
    
### ERROR AND CONFIGURATION CHECKS:

   if policy == 'normal_selective':
      bad_genoms_mutation_prob = 0.7
      
   elif policy == 'more_selective':
      bad_genoms_mutation_prob = 0.85
 
   elif policy == 'less_selective':
      bad_genoms_mutation_prob = 0.6
      
   else:
      raise ValueError("policy parameter must be: 'normal_selective' or 'more_selective' or 'less_selective'")
   
   if bad_genoms_mutation_prob is not None:
      if not isinstance(bad_genoms_mutation_prob, float) or bad_genoms_mutation_prob < 0 or bad_genoms_mutation_prob > 1:
         raise ValueError("bad_genoms_mutation_prob parameter must be float and 0-1 range")
      
   if activation_mutate_prob is not None:
      if not isinstance(activation_mutate_prob, float) or activation_mutate_prob < 0 or activation_mutate_prob > 1:
         raise ValueError("activation_mutate_prob parameter must be float and 0-1 range")
      
   if len(y_reward) % 2 == 0:
      slice_center = int(len(y_reward) / 2)

   else:
      raise ValueError("genom population size must be even number. for example: not 99, make 100 or 98.")

   sort_indices = np.argsort(y_reward)

### REWARD LIST IS SORTED IN ASCENDING ORDER, AND THE WEIGHT AND ACTIVATIONS OF EACH GENOME ARE SORTED ACCORDING TO THIS ORDER:

   y_reward = y_reward[sort_indices]
   weights = weights[sort_indices]

   activation_potentiations = [activation_potentiations[i] for i in sort_indices]

### GENOMES ARE DIVIDED INTO TWO GROUPS: GOOD GENOMES AND BAD GENOMES:

   best_weights = weights[slice_center:]
   bad_weights = weights[:slice_center]
   best_weight = best_weights[len(best_weights)-1]

   best_activations = list(activation_potentiations[slice_center:])
   bad_activations = list(activation_potentiations[:slice_center])
   best_activation = best_activations[len(best_activations) - 1]

   
### NEAT IS APPLIED ACCORDING TO THE SPECIFIED POLICY, STRATEGY, AND PROBABILITY CONFIGURATION:
   
   bar_format = loading_bars()[0]
   
   for i in tqdm(range(len(bad_weights)), desc="GENERATION: " + str(what_gen), bar_format=bar_format, ncols=50, ascii="▱▰"):

      if policy == 'normal_selective':
            
         if strategy == 'cross_over':
            bad_weights[i], bad_activations[i] = cross_over(best_weight, best_weights[i], best_activations=best_activation, good_activations=best_activations[i])


         elif strategy == 'potentiate':
            bad_weights[i], bad_activations[i] = potentiate(best_weight, best_weights[i], best_activations=best_activation, good_activations=best_activations[i])
            
         
         if mutations is True:
            
            mutation_prob = random.uniform(0, 1)

            if mutation_prob > bad_genoms_mutation_prob:
               if (save_best_genom == True and not np.array_equal(best_weights[i], best_weight)) or save_best_genom == False:
                  best_weights[i], best_activations[i] = mutation(best_weights[i], best_activations[i], activation_mutate_prob)
               
            elif mutation_prob < bad_genoms_mutation_prob:
               bad_weights[i], bad_activations[i] = mutation(bad_weights[i], bad_activations[i], activation_mutate_prob)

      if policy == 'more_selective':
               
            if strategy == 'cross_over':            
               bad_weights[i], bad_activations[i] = cross_over(best_weight, best_weights[i], best_activations=best_activation, good_activations=best_activations[i])
            
            elif strategy == 'potentiate':
               bad_weights[i], bad_activations[i] = potentiate(best_weight, best_weights[i], best_activations=best_activation, good_activations=best_activations[i])
            
            if mutations is True:

               mutation_prob = random.uniform(0, 1)

               if mutation_prob > bad_genoms_mutation_prob:
                  if (save_best_genom == True and not np.array_equal(best_weights[i], best_weight)) or save_best_genom == False:
                     best_weights[i], best_activations[i] = mutation(best_weights[i], best_activations[i], activation_mutate_prob)
                  
               elif mutation_prob < bad_genoms_mutation_prob:
                  bad_weights[i], bad_activations[i] = mutation(bad_weights[i], bad_activations[i], activation_mutate_prob)



      if policy == 'less_selective':

            random_index = int(random.uniform(0, len(best_weights) - 1))
            
            if strategy == 'cross_over': 
               bad_weights[i], bad_activations[i] = cross_over(best_weights[random_index], best_weights[i], best_activations=best_activations[random_index], good_activations=best_activations[i])
            
            elif strategy == 'potentiate':
               bad_weights[i], bad_activations[i] = potentiate(best_weights[random_index], best_weights[i], best_activations=best_activations[random_index], good_activations=best_activations[i])
            
            if mutations is True:

               mutation_prob = random.uniform(0, 1)

               if mutation_prob > bad_genoms_mutation_prob:
                  if (save_best_genom == True and not np.array_equal(best_weights[i], best_weight)) or save_best_genom == False:
                     best_weights[i], best_activations[i] = mutation(best_weights[i], best_activations[i], activation_mutate_prob)
                  
               elif mutation_prob < bad_genoms_mutation_prob:
                  bad_weights[i], bad_activations[i] = mutation(bad_weights[i], bad_activations[i], activation_mutate_prob)


   weights = np.vstack((bad_weights, best_weights))
   activation_potentiations = bad_activations + best_activations

   ### INFO PRINTING CONSOLE
   
   if show_info == True:
      print("\nGENERATION:", str(what_gen) + ' IS FINISHED \n')
      print("*** Configuration Settings ***")
      print("  STRATEGY: ", strategy)
      print("  POLICY: ", policy)
      print("  MUTATIONS: ", str(mutations))
      print("  BAD GENOMES MUTATION PROB: ", str(round(bad_genoms_mutation_prob, 2)))
      print("  GOOD GENOMES MUTATION PROB: ", str(round(1 - bad_genoms_mutation_prob, 2)) + '\n')
   
      print("*** Performance ***")
      print("  MAX REWARD: ", str(round(max(y_reward), 2)))
      print("  MEAN REWARD: ", str(round(np.mean(y_reward), 2)))
      print("  MIN REWARD: ", str(round(min(y_reward), 2)) + '\n')
      

   return np.array(weights), activation_potentiations


def evaluate(x_population, weights, activation_potentiations, rl_mode=False):
   """
    Evaluates the performance of a population of genomes, applying different activation functions 
    and weights depending on whether reinforcement learning mode is enabled or not.

    Args:
        x_population (list or numpy.ndarray): A list or 2D numpy array where each element represents
                                               a genome (A list of input features for each genome, or a single set of input features for one genome (only in rl_mode)).
        weights (list or numpy.ndarray): A list or 2D numpy array of weights corresponding to each genome 
                                         in `x_population`. This determines the strength of connections.
        activation_potentiations (list or str): A list where each entry represents an activation function 
                                                or a potentiation strategy applied to each genome. If only one 
                                                activation function is used, this can be a single string.
        rl_mode (bool, optional): If True, reinforcement learning mode is activated, this accepts x_population is a single genom. (Also weights and activation_potentations a single genomes part.)
                                  Default is False.

    Returns:
        list: A list of outputs corresponding to each genome in the population after applying the respective 
              activation function and weights.

    Notes:
        - If `rl_mode` is True:
            - Accepts x_population is a single genom
            - The inputs are flattened, and the activation function is applied across the single genom.
        
        - If `rl_mode` is False:
            - Accepts x_population is a list of genomes
            - Each genome is processed individually, and the results are stored in the `outputs` list.
        
        - `fex()` function is the core function that processes the input with the given weights and activation function.
    
    Example:
        ```python
        outputs = evaluate(x_population, weights, activation_potentiations, rl_mode=False)
        ```

    - The function returns a list of outputs after processing the population, where each element corresponds to 
      the output for each genome in `x_population`.
   """
   
   ### IF RL_MODE IS TRUE, A SINGLE GENOME IS ASSUMED AS INPUT, A FEEDFORWARD PREDICTION IS MADE, AND THE OUTPUT(NPARRAY) IS RETURNED:
   
   ### IF RL_MODE IS FALSE, PREDICTIONS ARE MADE FOR ALL GENOMES IN THE GROUP USING THEIR CORRESPONDING INDEXED INPUTS AND DATA.
   ### THE OUTPUTS ARE RETURNED AS A PYTHON LIST, WHERE EACH GENOME'S OUTPUT MATCHES ITS INDEX:
   
   if rl_mode == True:
      Input = np.array(x_population)
      Input = Input.ravel()
      
      if isinstance(activation_potentiations, str):
         activation_potentiations = [activation_potentiations]
      
      outputs = fex(Input=Input, is_training=False, activation_potentiation=activation_potentiations, w=weights)
      
   else:
      outputs = [0] * len(x_population)
      for i, genome in enumerate(x_population):

         Input = np.array(genome)
         Input = Input.ravel()

         if isinstance(activation_potentiations[i], str):
            activation_potentiations[i] = [activation_potentiations[i]]

         outputs[i] = fex(Input=Input, is_training=False, activation_potentiation=activation_potentiations[i], w=weights[i])

   return outputs


def cross_over(best_weight, good_weight, best_activations, good_activations):
   """
    Performs a Two-Point Matrix Crossover operation on two sets of weights and activation functions.
    This function combines two individuals (represented by their weights and activation functions) 
    to create a new individual by exchanging parts of their weight matrices and activation functions.

    Args:
        best_weight (numpy.ndarray): The weight matrix of the first individual (parent).
        good_weight (numpy.ndarray): The weight matrix of the second individual (parent).
        best_activations (str or list): The activation function(s) of the first individual.
        good_activations (str or list): The activation function(s) of the second individual.

    Returns:
        tuple: A tuple containing:
            - new_weight (numpy.ndarray): The weight matrix of the new individual created by crossover.
            - new_activations (list): The list of activation functions of the new individual created by crossover.

    Notes:
        - The crossover is performed by selecting random sub-matrices from the parent weight matrices 
          and swapping them.
        - The crossover operation combines the activation functions of both parents by concatenating them.
        - If the activation functions are passed as strings, they are converted to lists for uniform handling.
   """
   
   ### THE GIVEN GENOMES' WEIGHTS ARE RANDOMLY SELECTED AND COMBINED OVER A RANDOM RANGE. SIMILARLY, THEIR ACTIVATIONS ARE COMBINED. A NEW GENOME IS RETURNED WITH THE COMBINED WEIGHTS FIRST, FOLLOWED BY THE ACTIVATIONS:
   
   start = 0
   
   row_end = best_weight.shape[0]
   col_end = best_weight.shape[1]

   while True:

      row_cut_start = int(random.uniform(start, row_end))
      col_cut_start = int(random.uniform(start, col_end))

      row_cut_end = int(random.uniform(start, row_end))
      col_cut_end = int(random.uniform(start, col_end))

      if (row_cut_end > row_cut_start) and (col_cut_end > col_cut_start):
         break

   new_weight = np.copy(best_weight)
   best_w2 = np.copy(good_weight)
   
   new_weight[row_cut_start:row_cut_end, col_cut_start:col_cut_end] = best_w2[row_cut_start:row_cut_end, col_cut_start:col_cut_end]

   if isinstance(best_activations, str):
      best = [best_activations]

   if isinstance(good_activations, str):
      good = [good_activations]

   if isinstance(best_activations, list):
      best = best_activations

   if isinstance(good_activations, list):
      good = good_activations

   new_activations = best + good

   return new_weight, new_activations

def potentiate(best_weight, good_weight, best_activations, good_activations):
   """
    Combines two sets of weights and activation functions by adding the weight matrices and 
    concatenating the activation functions. The resulting weight matrix is normalized. (Max abs normalization.)
    
    Args:
        best_weight (numpy.ndarray): The weight matrix of the first individual (parent).
        good_weight (numpy.ndarray): The weight matrix of the second individual (parent).
        best_activations (str or list): The activation function(s) of the first individual.
        good_activations (str or list): The activation function(s) of the second individual.
    
    Returns:
        tuple: A tuple containing:
            - new_weight (numpy.ndarray): The new weight matrix after potentiation and normalization. (Max abs normalization.)
            - new_activations (list): The new activation functions after concatenation.
    
    Notes:
        - The weight matrices are element-wise added and then normalized using the `normalization` function. (Max abs normalization.)
        - The activation functions from both parents are concatenated to form the new activation functions list.
        - If the activation functions are passed as strings, they are converted to lists for uniform handling.
   """
    
   new_weight = best_weight + good_weight
   new_weight = normalization(new_weight)
   
   if isinstance(best_activations, str):
      best = [best_activations]

   if isinstance(good_activations, str):
      good = [good_activations]

   if isinstance(best_activations, list):
      best = best_activations

   if isinstance(good_activations, list):
      good = good_activations
   
   new_activations = best + good
   
   return new_weight, new_activations

def mutation(weight, activations, activation_mutate_prob):
   """
    Performs mutation on the given weight matrix and activations list.
    - The weight matrix is mutated by randomly changing its values.
    - The activation functions are mutated by adding, removing, or replacing them.

    Args:
        weight (numpy.ndarray): The weight matrix to mutate.
        activations (list): The list of activation functions to mutate.
        activation_mutate_prob (float): Probability of mutation occurring on the activation functions.
        activations_list (list): A predefined list of possible activation functions to choose from.

    Returns:
        tuple: A tuple containing:
            - mutated weight matrix (numpy.ndarray)
            - mutated activation functions (list)
   """
   start = 0
   row_end = weight.shape[0]
   col_end = weight.shape[1]
   threshold = 32
   new_threshold = threshold
   
   while True:
      
      selected_row = int(random.uniform(start, row_end))
      selected_col = int(random.uniform(start, col_end))

      weight[selected_row, selected_col] = random.uniform(-1, 1)

      if int(row_end * col_end) > new_threshold:
         new_threshold += threshold
         pass
      
      else:
         break 
   
   
   activation_mutate_prob = 1 - activation_mutate_prob # if prob 0.8 (%80) then 1 - 0.8. Because 0-1 random number probably greater than 0.2
   potential_activation_mutation = random.uniform(0, 1)

   if potential_activation_mutation > activation_mutate_prob:
      all_activations = activations_list()
      
      mutation_action_prob = random.uniform(0, 1)

      if mutation_action_prob < 0.33:

         try:
            
            random_index_all_act = int(random.uniform(0, len(all_activations)-1))
            activations.append(all_activations[random_index_all_act])

         except:

            activation = activations
            activations = []

            activations.append(activation)
            activations.append(all_activations[int(random.uniform(0, len(all_activations)-1))])
            
         if mutation_action_prob < 0.66 and mutation_action_prob > 0.33 and len(activations) > 1:
            
            random_index = int(random.uniform(0, len(activation)-1))
            activations.remove(random_index)
            
         if mutation_action_prob > 0.66 and len(activations) > 1:
            
            random_index_all_act = int(random.uniform(0, len(all_activations)-1))
            random_index_genom_act = int(random.uniform(0, len(activations)-1))
            
            activations[random_index_genom_act] = all_activations[random_index_all_act]
            

   return weight, activations