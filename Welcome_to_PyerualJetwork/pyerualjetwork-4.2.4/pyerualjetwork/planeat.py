""" 
MAIN MODULE FOR PLANEAT

Examples: https://github.com/HCB06/PyerualJetwork/tree/main/Welcome_to_PyerualJetwork/ExampleCodes

ANAPLAN document: https://github.com/HCB06/Anaplan/blob/main/Welcome_to_Anaplan/ANAPLAN_USER_MANUEL_AND_LEGAL_INFORMATION(EN).pdf

@author: Hasan Can Beydili
@YouTube: https://www.youtube.com/@HasanCanBeydili
@Linkedin: https://www.linkedin.com/in/hasan-can-beydili-77a1b9270/
@Instagram: https://www.instagram.com/canbeydili.06/
@contact: tchasancan@gmail.com
"""

import numpy as np
import random
import math

### LIBRARY IMPORTS ###
from .plan import feed_forward
from .data_operations import normalization
from .ui import loading_bars, initialize_loading_bar
from .activation_functions import apply_activation, all_activations

def define_genomes(input_shape, output_shape, population_size, dtype=np.float32):
   """
   Initializes a population of genomes, where each genome is represented by a set of weights 
   and an associated activation function. Each genome is created with random weights and activation 
   functions are applied and normalized. (Max abs normalization.)

   Args:

      input_shape (int): The number of input features for the neural network.

      output_shape (int): The number of output features for the neural network.
      
      population_size (int): The number of genomes (individuals) in the population.

      dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)

   Returns:
      tuple: A tuple containing:
         - population_weights (numpy.ndarray): A 2D numpy array of shape (population_size, output_shape, input_shape) representing the 
            weight matrices for each genome.
         - population_activations (list): A list of activation functions applied to each genome.
            
   Notes:
      The weights are initialized randomly within the range [-1, 1]. 
      Activation functions are selected randomly from a predefined list `all_activations()`.
      The weights for each genome are then modified by applying the corresponding activation function 
      and normalized using the `normalization()` function. (Max abs normalization.)
   """
   population_weights = [0] * population_size
   population_activations = [0] * population_size

   except_this = ['spiral', 'circular']
   activations = [item for item in all_activations() if item not in except_this] # SPIRAL AND CIRCULAR ACTIVATION DISCARDED

   for i in range(len(population_weights)):

      population_weights[i] = np.random.uniform(-1, 1, (output_shape, input_shape)).astype(dtype)
      population_activations[i] = activations[int(random.uniform(0, len(activations)-1))]

      # ACTIVATIONS APPLYING IN WEIGHTS SPECIFIC OUTPUT CONNECTIONS (MORE PLAN LIKE FEATURES(FOR NON-LINEARITY)):

      for j in range(population_weights[i].shape[0]):

         population_weights[i][j,:] = apply_activation(population_weights[i][j,:], population_activations[i])
         population_weights[i][j,:] = normalization(population_weights[i][j,:], dtype=dtype)

   return np.array(population_weights, dtype=dtype), population_activations


def evolver(weights, 
            activation_potentiations, 
            what_gen, 
            fitness, 
            show_info=False, 
            policy='aggressive', 
            bad_genomes_selection_prob=None, 
            bar_status=True, 
            strategy='normal_selective', 
            target_fitness='max', 
            bad_genomes_mutation_prob=None, 
            activation_mutate_prob=0.5, 
            save_best_genome=True, 
            fitness_bias=None, 
            cross_over_mode='tpm', 
            activation_mutate_add_prob=0.5, 
            activation_mutate_delete_prob=0.5, 
            activation_mutate_change_prob=0.5, 
            weight_mutate_prob=1, 
            weight_mutate_rate=32, 
            activation_selection_add_prob=0.6, 
            activation_selection_change_prob=0.4, 
            activation_selection_rate=2, 
            dtype=np.float32):
   """
   Applies the evolving process of a population of genomes using selection, crossover, mutation, and activation function potentiation.
   The function modifies the population's weights and activation functions based on a specified policy, mutation probabilities, and strategy.

   'selection' args effects cross-over.
   'mutate' args effects mutation.

   Args:
      weights (numpy.ndarray): Array of weights for each genome. 
         (first returned value of define_genomes function)
      
      activation_potentiations (list): A list of activation functions for each genome. 
         (second returned value of define_genomes function)
      
      what_gen (int): The current generation number, used for informational purposes or logging.
      
      fitness (numpy.ndarray): A 1D array containing the fitness values of each genome. 
         The array is used to rank the genomes based on their performance. PLANEAT maximizes or minimizes this fitness based on the `target_fitness` parameter.
      
      show_info (bool, optional): If True, prints information about the current generation and the 
         maximum reward obtained. Also shows the current configuration. Default is False.
      
      strategy (str, optional): The strategy for combining the best and bad genomes. Options:
         - 'normal_selective': Normal selection based on reward, where a portion of the bad genes are discarded.
         - 'more_selective': A more selective strategy, where fewer bad genes survive.
         - 'less_selective': A less selective strategy, where more bad genes survive.
         Default is 'normal_selective'.
      
      bar_status (bool, optional): Loading bar status during evolving process of genomes. True or False. Default: True

      policy (str, optional): The selection policy that governs how genomes are selected for reproduction. Options:
            
            - 'aggressive': Aggressive policy using very aggressive selection policy. 
                  Advantages: fast training. 
                  Disadvantages: may lead to fitness stuck in a local maximum or minimum.

            - 'explorer': Explorer policy increases population diversity.
                  Advantages: fitness does not get stuck at local maximum or minimum.
                  Disadvantages: slow training.
                  
                  Suggestions: Use hybrid and dynamic policy. When fitness appears stuck, switch to the 'explorer' policy.

            Default: 'aggressive'.
      
      target_fitness (str, optional): Target fitness strategy for PLANEAT optimization. ('max' maximizes fitness, 'min' minimizes fitness.) Default: 'max'.
      
      fitness_bias (float, optional): Fitness bias must be a probability value between 0 and 1 that determines the effect of fitness on the crossover process. Default: Determined by the `strategy`.
      
      bad_genomes_mutation_prob (float, optional): The probability of applying mutation to the bad genomes. 
         Must be in the range [0, 1]. Also affects the mutation probability of the best genomes inversely. 
         For example, a value of 0.7 for bad genomes implies 0.3 for best genomes. Default: Determined by `policy`.
      
      activation_mutate_prob (float, optional): The probability of applying mutation to the activation functions. 
         Must be in the range [0, 1]. Default is 0.5 (50%).
      
      save_best_genom (bool, optional): If True, ensures that the best genomes are saved and not mutated 
         or altered during reproduction. Default is True.
      
      cross_over_mode (str, optional): Specifies the crossover method to use. Options:
         - 'tpm': Two-Point Matrix Crossover.
         Default is 'tpm'.
      
      activation_mutate_add_prob (float, optional): The probability of adding a new activation function to the genome for mutation. 
         Must be in the range [0, 1]. Default is 0.5.
      
      activation_mutate_delete_prob (float, optional): The probability of deleting an existing activation function 
         from the genome for mutation. Must be in the range [0, 1]. Default is 0.5.
      
      activation_mutate_change_prob (float, optional): The probability of changing an activation function in the genome for mutation. 
         Must be in the range [0, 1]. Default is 0.5.
      
      weight_mutate_prob (float, optional): The probability of mutating a weight in the genome. 
         Must be in the range [0, 1]. Default is 1.
      
      weight_mutate_rate (int, optional): If the value entered here equals the result of input_layer * output_layer, 
         only a single weight will be mutated during each mutation process. If the value is half of the result, 
         two weights will be mutated. WARNING: If you don't understand, do NOT change this value. Default is 32.

      activation_selection_add_prob (float, optional): The probability of adding an existing activation function for crossover.
         Must be in the range [0, 1]. Default is 0.6. (WARNING! Higher values increase complexity. For faster training, increase this value.)
      
      activation_selection_change_prob (float, optional): The probability of changing an activation function in the genome for crossover. 
         Must be in the range [0, 1]. Default is 0.4.
      
      activation_selection_rate (int, optional): If the activation list of a good genome is smaller than this value, only one activation will undergo crossover. This parameter controls model complexity. Default is 2.

      dtype (numpy.dtype): Data type for the arrays. Default: np.float32. 
         Example: np.float64 or np.float16 [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not recommended!].

   Raises:
      ValueError: 
         - If `policy` is not one of the specified values ('aggressive', 'explorer').
         - If 'strategy' is not one of the specified values ('less_selective', 'normal_selective', 'more_selective')
         - If `cross_over_mode` is not one of the specified values ('tpm').
         - If `bad_genomes_mutation_prob`, `activation_mutate_prob`, or other probability parameters are not in the range 0 and 1.
         - If the population size is odd (ensuring an even number of genomes is required for proper selection).
         - If 'fitness_bias' value is not in range 0 and 1.

   Returns:
      tuple: A tuple containing:
         - weights (numpy.ndarray): The updated weights for the population after selection, crossover, and mutation. 
                                    The shape is (population_size, output_shape, input_shape).
         - activation_potentiations (list): The updated list of activation functions for the population.

   Notes:
      - **Selection Process**: 
         - The genomes are sorted by their fitness (based on `fitness`), and then split into "best" and "bad" halves. 
         - The best genomes are retained, and the bad genomes are modified based on the selected strategy.
         
      - **Crossover Strategies**:
         - The **'cross_over'** strategy performs crossover, where parts of the best genomes' weights are combined with other good genomes to create new weight matrices.
         
      - **Mutation**:
         - Mutation is applied to both the best and bad genomes, depending on the mutation probability and the `policy`.
         - `bad_genomes_mutation_prob` determines the probability of applying mutations to the bad genomes.
         - If `activation_mutate_prob` is provided, activation function mutations are applied to the genomes based on this probability.
         
      - **Population Size**: The population size must be an even number to properly split the best and bad genomes. If `fitness` has an odd length, an error is raised.
      
      - **Logging**: If `show_info=True`, the current generation and the maximum reward from the population are printed for tracking the learning progress.

   Example:
      ```python
      weights, activation_potentiations = planeat.evolver(weights, activation_potentiations, 1, fitness, show_info=True, strategy='normal_selective', policy='aggressive')
      ```

      - The function returns the updated weights and activations after processing based on the chosen strategy, policy, and mutation parameters.
   """

### ERROR AND CONFIGURATION CHECKS:

   if strategy == 'normal_selective':
      if bad_genomes_mutation_prob is None: bad_genomes_mutation_prob = 0.7 # EFFECTS MUTATION
      if bad_genomes_selection_prob is None: bad_genomes_selection_prob = 0.25 # EFFECTS CROSS-OVER
      if fitness_bias is None: fitness_bias = 0.5 # The pressure applied by FITNESS to the CROSS-OVER
      
   elif strategy == 'more_selective':
      if bad_genomes_mutation_prob is None: bad_genomes_mutation_prob = 0.85 # EFFECTS MUTATION
      if bad_genomes_selection_prob is None: bad_genomes_selection_prob = 0.1 # EFFECTS CROSS-OVER
      if fitness_bias is None: fitness_bias = 0.7 # The pressure applied by FITNESS to the CROSS-OVER

   elif strategy == 'less_selective':
      if bad_genomes_mutation_prob is None: bad_genomes_mutation_prob = 0.6 # EFFECTS MUTATION
      if bad_genomes_selection_prob is None: bad_genomes_selection_prob = 0.5 # EFFECTS CROSS-OVER
      if fitness_bias is None: fitness_bias = 0.3 # The pressure applied by FITNESS to the CROSS-OVER

   else:
      raise ValueError("strategy parameter must be: 'normal_selective' or 'more_selective' or 'less_selective'")

   if ((activation_mutate_add_prob < 0 or activation_mutate_add_prob > 1) or 
      (activation_mutate_change_prob < 0 or activation_mutate_change_prob > 1) or 
      (activation_mutate_delete_prob < 0 or activation_mutate_delete_prob > 1) or 
      (weight_mutate_prob < 0 or weight_mutate_prob > 1) or 
      (activation_selection_add_prob < 0 or activation_selection_add_prob > 1) or (
      activation_selection_change_prob < 0 or activation_selection_change_prob > 1)):
      
      raise ValueError("All hyperparameters ending with 'prob' must be a number between 0 and 1.")
   
   if fitness_bias < 0 or fitness_bias > 1: raise ValueError("fitness_bias value must be a number between 0 and 1.")

   if bad_genomes_mutation_prob is not None:
      if not isinstance(bad_genomes_mutation_prob, float) or bad_genomes_mutation_prob < 0 or bad_genomes_mutation_prob > 1:
         raise ValueError("bad_genomes_mutation_prob parameter must be float and 0-1 range")
      
   if activation_mutate_prob is not None:
      if not isinstance(activation_mutate_prob, float) or activation_mutate_prob < 0 or activation_mutate_prob > 1:
         raise ValueError("activation_mutate_prob parameter must be float and 0-1 range")
      
   if len(fitness) % 2 == 0:
      slice_center = int(len(fitness) / 2)

   else:
      raise ValueError("genome population size must be even number. for example: not 99, make 100 or 98.")

   
### FITNESS IS SORTED IN ASCENDING (OR DESCENDING) ORDER, AND THE WEIGHT AND ACTIVATIONS OF EACH GENOME ARE SORTED ACCORDING TO THIS ORDER:

   if target_fitness == 'max': sort_indices = np.argsort(fitness)
   elif target_fitness == 'min': sort_indices = np.argsort(-fitness)

   fitness = fitness[sort_indices]
   weights = weights[sort_indices]

   activation_potentiations = [activation_potentiations[i] for i in sort_indices]

### GENOMES ARE DIVIDED INTO TWO GROUPS: GOOD GENOMES AND BAD GENOMES:

   good_weights = weights[slice_center:]
   bad_weights = weights[:slice_center]
   best_weight = good_weights[-1]

   good_activations = list(activation_potentiations[slice_center:])
   bad_activations = list(activation_potentiations[:slice_center])
   best_activations = good_activations[-1]

   
### PLANEAT IS APPLIED ACCORDING TO THE SPECIFIED POLICY, STRATEGY, AND PROBABILITY CONFIGURATION:
   
   bar_format = loading_bars()[0]
   
   if bar_status: progress = initialize_loading_bar(len(bad_weights), desc="GENERATION: " + str(what_gen), bar_format=bar_format, ncols=50)
   normalized_fitness = abs(normalization(fitness, dtype=dtype))

   best_fitness = normalized_fitness[-1]
   epsilon = np.finfo(float).eps

   child_W = np.copy(bad_weights)
   child_act = bad_activations.copy()

   mutated_W = np.copy(bad_weights)
   mutated_act = bad_activations.copy()

   for i in range(len(bad_weights)):
      
      if policy == 'aggressive':
         first_parent_W = best_weight
         first_parent_act = best_activations

      elif policy == 'explorer':
         first_parent_W = good_weights[i]
         first_parent_act = good_activations[i]

      else: raise ValueError("policy parameter must be: 'aggressive' or 'explorer'")
         
      second_parent_W, second_parent_act, s_i = second_parent_selection(good_weights, bad_weights, good_activations, bad_activations, bad_genomes_selection_prob)

      child_W[i], child_act[i] = cross_over(first_parent_W,
                                             second_parent_W,
                                             first_parent_act,
                                             second_parent_act,
                                             cross_over_mode=cross_over_mode,
                                             activation_selection_add_prob=activation_selection_add_prob,
                                             activation_selection_change_prob=activation_selection_change_prob,
                                             threshold=activation_selection_rate,
                                             bad_genomes_selection_prob=bad_genomes_selection_prob,
                                             first_parent_fitness=best_fitness,
                                             fitness_bias=fitness_bias,
                                             second_parent_fitness=normalized_fitness[s_i],
                                             epsilon=epsilon
                                             )

      mutation_prob = random.uniform(0, 1)

      if mutation_prob > bad_genomes_mutation_prob:
            genome_W = good_weights[i]
            genome_act = good_activations[i]

            fitness_index = int(len(bad_weights) / 2 + i)

      else:
            genome_W = bad_weights[i]
            genome_act = bad_activations[i]

            fitness_index = i

      mutated_W[i], mutated_act[i] = mutation(genome_W, 
                                             genome_act,
                                             activation_mutate_prob=activation_mutate_prob,
                                             activation_add_prob=activation_mutate_add_prob,
                                             activation_delete_prob=activation_mutate_delete_prob, 
                                             activation_change_prob=activation_mutate_change_prob, 
                                             weight_mutate_prob=weight_mutate_prob, 
                                             threshold=weight_mutate_rate,
                                             genome_fitness=normalized_fitness[fitness_index]
                                             )

      if bar_status: progress.update(1)

   if save_best_genome:
      child_W[-1] = best_weight
      child_act[-1] = best_activations

   weights = np.vstack((child_W, mutated_W))
   activation_potentiations = child_act + mutated_act

   ### INFO PRINTING CONSOLE
   
   if show_info == True:
      print("\nGENERATION:", str(what_gen) + ' FINISHED \n')
      print("*** Configuration Settings ***")
      print("  POPULATION SIZE: ", str(len(weights)))
      print("  STRATEGY: ", strategy)
      print("  CROSS OVER MODE: ", cross_over_mode)
      print("  POLICY: ", policy)
      print("  BAD GENOMES MUTATION PROB: ", str(bad_genomes_mutation_prob))
      print("  GOOD GENOMES MUTATION PROB: ", str(round(1 - bad_genomes_mutation_prob, 2)))
      print("  BAD GENOMES SELECTION PROB: ", str(bad_genomes_selection_prob))
      print("  WEIGHT MUTATE PROB: ", str(weight_mutate_prob))
      print("  WEIGHT MUTATE RATE (THRESHOLD VALUE FOR SINGLE MUTATION): ", str(weight_mutate_rate))
      print("  ACTIVATION MUTATE PROB: ", str(activation_mutate_prob))
      print("  ACTIVATION MUTATE ADD PROB: ", str(activation_mutate_add_prob))
      print("  ACTIVATION MUTATE DELETE PROB: ", str(activation_mutate_delete_prob))
      print("  ACTIVATION MUTATE CHANGE PROB: ", str(activation_mutate_change_prob))
      print("  ACTIVATION SELECTION ADD PROB: ", str(activation_selection_add_prob))
      print("  ACTIVATION SELECTION CHANGE PROB: ", str(activation_selection_change_prob))
      print("  FITNESS BIAS: ", str(fitness_bias))
      print("  ACTIVATION SELECTION RATE (THRESHOLD VALUE FOR SINGLE CROSS OVER):", str(activation_selection_rate) + '\n')


      print("*** Performance ***")
      print("  MAX FITNESS: ", str(round(max(fitness), 2)))
      print("  MEAN FITNESS: ", str(round(np.mean(fitness), 2)))
      print("  MIN FITNESS: ", str(round(min(fitness), 2)) + '\n')

      print("  BEST GENOME ACTIVATION LENGTH: ", str(len(activation_potentiations[-1])))
      print("  BEST GENOME INDEX: ", str(len(weights)-1))
      print("  NOTE: Genomes are always sorted from the least successful to the most successful according to their performance ranking. Therefore, the genome at the last index is the king of the previous generation. " + '\n')
      

   return weights, activation_potentiations


def evaluate(x_population, weights, activation_potentiations, rl_mode=False, dtype=np.float32):
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
        rl_mode (bool, optional): If True, reinforcement learning mode is activated, this accepts x_population is a single genome. (Also weights and activation_potentations a single genomes part.)
                                  Default is False.

        dtype (numpy.dtype): Data type for the arrays. np.float32 by default. Example: np.float64 or np.float16. [fp32 for balanced devices, fp64 for strong devices, fp16 for weak devices: not reccomended!] (optional)                         
         
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
        
        - `feed_forward()` function is the core function that processes the input with the given weights and activation function.
    
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
      Input = np.array(x_population, copy=False, dtype=dtype)
      Input = Input.ravel()
      
      if isinstance(activation_potentiations, str):
         activation_potentiations = [activation_potentiations]
      
      outputs = feed_forward(Input=Input, is_training=False, activation_potentiation=activation_potentiations, w=weights)
      
   else:
      outputs = [0] * len(x_population)
      for i, genome in enumerate(x_population):

         Input = np.array(genome, copy=False)
         Input = Input.ravel()

         if isinstance(activation_potentiations[i], str):
            activation_potentiations[i] = [activation_potentiations[i]]

         outputs[i] = feed_forward(Input=Input, is_training=False, activation_potentiation=activation_potentiations[i], w=weights[i])

   return outputs


def cross_over(first_parent_W,
               second_parent_W,
               first_parent_act,
               second_parent_act,
               cross_over_mode,
               activation_selection_add_prob,
               activation_selection_change_prob,
               threshold,
               bad_genomes_selection_prob,
               first_parent_fitness,
               second_parent_fitness,
               fitness_bias,
               epsilon):
   """
    Performs a crossover operation on two sets of weights and activation functions.
    This function combines two individuals (represented by their weights and activation functions) 
    to create a new individual by exchanging parts of their weight matrices and activation functions.

    Args:
        first_parent_W (numpy.ndarray): The weight matrix of the first individual (parent).
        
        second_parent_W (numpy.ndarray): The weight matrix of the second individual (parent).
        
        first_parent_act (str or list): The activation function(s) of the first individual.
        
        second_parent_act (str or list): The activation function(s) of the second individual.
        
        cross_over_mode (str): Determines the crossover method to be used. Options:
            - 'tpm': Two-Point Matrix Crossover, where sub-matrices of weights are swapped between parents.
        
        activation_selection_add_prob (float): Probability of adding new activation functions 
            from the second parent to the child genome.
        
        activation_selection_change_prob (float): Probability of replacing an activation function in the child genome 
            with one from the second parent.
        
        threshold (float): Determines how quickly activation functions are added or replaced 
            during the crossover process.
        
        bad_genomes_selection_prob (float): Probability of selecting a "bad" genome for replacement with the offspring.
        
        first_parent_fitness (float): Fitness score of the first parent.
        
        second_parent_fitness (float): Fitness score of the second parent.
        
        fitness_bias (float): A bias factor used to favor fitter parents during crossover operations.

        epsilon (float): Small epsilon constant

    Returns:
        tuple: A tuple containing:
            - child_W (numpy.ndarray): The weight matrix of the new individual created by crossover.
            - child_act (list): The list of activation functions of the new individual created by crossover.

    Notes:
        - The crossover is performed based on the selected `cross_over_mode`.
        - In 'tpm' mode, random sub-matrices from the parent weight matrices are swapped.
        - Activation functions from both parents are combined using the probabilities and rates provided.

    Example:
        ```python
        new_weights, new_activations = cross_over(
            first_parent_W=parent1_weights, 
            second_parent_W=parent2_weights, 
            first_parent_act=parent1_activations, 
            second_parent_act=parent2_activations, 
            cross_over_mode='tpm',
            activation_selection_add_prob=0.8,
            activation_selection_change_prob=0.5,
            threshold=2,
            bad_genomes_selection_prob=0.7,
            first_parent_fitness=0.9,
            second_parent_fitness=0.85,
            fitness_bias=0.6,
            epsilon=np.finfo(float).eps
        )
        ```
    """
   
   ### THE GIVEN GENOMES' WEIGHTS ARE RANDOMLY SELECTED AND COMBINED OVER A RANDOM RANGE. SIMILARLY, THEIR ACTIVATIONS ARE COMBINED. A NEW GENOME IS RETURNED WITH THE COMBINED WEIGHTS FIRST, FOLLOWED BY THE ACTIVATIONS:
   
   start = 0
   
   row_end = first_parent_W.shape[0]
   col_end = first_parent_W.shape[1]

   total_gene = row_end * col_end
   half_of_gene = int(total_gene / 2)

   decision = dominant_parent_selection(bad_genomes_selection_prob)

   if decision == 'first_parent':
      dominant_parent_W = np.copy(first_parent_W)
      dominant_parent_act = first_parent_act

      undominant_parent_W = np.copy(second_parent_W)
      undominant_parent_act = second_parent_act
      succes = second_parent_fitness + epsilon

   elif decision == 'second_parent':
      dominant_parent_W = np.copy(second_parent_W)
      dominant_parent_act = second_parent_act

      undominant_parent_W = np.copy(first_parent_W)
      undominant_parent_act = first_parent_act
      succes = first_parent_fitness + epsilon

   while True:

      row_cut_start = int(random.uniform(start, row_end))
      col_cut_start = int(random.uniform(start, col_end))

      row_cut_end = int(random.uniform(start, row_end))
      col_cut_end = int(random.uniform(start, col_end))

      if ((row_cut_end > row_cut_start) and
         (col_cut_end > col_cut_start) and
         (((row_cut_end + 1) - (row_cut_start + 1) * 2) + ((col_cut_end + 1) - (col_cut_start + 1) * 2) <= half_of_gene)):
         break
      
      selection_bias = random.uniform(0, 1)

      if fitness_bias > selection_bias:
         row_cut_start = math.floor(row_cut_start * (succes + epsilon))
         row_cut_end = math.ceil(row_cut_end * (succes + epsilon))

         col_cut_start = math.floor(col_cut_start * (succes + epsilon))
         col_cut_end = math.ceil(col_cut_end * (succes + epsilon))

   child_W = dominant_parent_W

   if cross_over_mode == 'tpm':
      child_W[row_cut_start:row_cut_end, col_cut_start:col_cut_end] = undominant_parent_W[row_cut_start:row_cut_end, col_cut_start:col_cut_end]

   if isinstance(dominant_parent_act, str): dominant_parent_act = [dominant_parent_act]
   if isinstance(undominant_parent_act, str): undominant_parent_act = [undominant_parent_act]

   child_act = list(np.copy(dominant_parent_act))

   activation_selection_add_prob = 1 - activation_selection_add_prob # if prob 0.8 (%80) then 1 - 0.8. Because 0-1 random number probably greater than 0.2
   potential_activation_selection_add = random.uniform(0, 1)

   if potential_activation_selection_add > activation_selection_add_prob:
      
      threshold = threshold / succes
      new_threshold = threshold
      
      while True:
         
         random_index = int(random.uniform(0, len(undominant_parent_act)-1))
         random_undominant_activation = undominant_parent_act[random_index]

         child_act.append(random_undominant_activation)

         if len(dominant_parent_act) > new_threshold:
            new_threshold += threshold
            pass
         
         else:
            break

   activation_selection_change_prob = 1 - activation_selection_change_prob
   potential_activation_selection_change_prob = random.uniform(0, 1)

   if potential_activation_selection_change_prob > activation_selection_change_prob:
      
      threshold = threshold / succes
      new_threshold = threshold
      
      while True:
         
         random_index_undominant = int(random.uniform(0, len(undominant_parent_act)-1))
         random_index_dominant = int(random.uniform(0, len(dominant_parent_act)-1))
         random_undominant_activation = undominant_parent_act[random_index_undominant]

         child_act[random_index_dominant] = random_undominant_activation

         if len(dominant_parent_act) > new_threshold:
            new_threshold += threshold
            pass
         
         else:
            break

   return child_W, child_act

def mutation(weight, 
             activations, 
             activation_mutate_prob, 
             activation_add_prob, 
             activation_delete_prob, 
             activation_change_prob, 
             weight_mutate_prob, 
             threshold, 
             genome_fitness
             ):
   """
    Performs mutation on the given weight matrix and activation functions.
    - The weight matrix is mutated by randomly changing its values based on the mutation probability.
    - The activation functions are mutated by adding, removing, or replacing them with predefined probabilities.

    Args:
        weight (numpy.ndarray): The weight matrix to mutate.
        
        activations (list): The list of activation functions to mutate.
        
        activation_mutate_prob (float): The overall probability of mutating activation functions.
        
        activation_add_prob (float): Probability of adding a new activation function.
        
        activation_delete_prob (float): Probability of removing an existing activation function.
        
        activation_change_prob (float): Probability of replacing an existing activation function with a new one.
        
        weight_mutate_prob (float): The probability of mutating weight matrix.
        
        threshold (float): If the value you enter here is equal to the result of input layer * output layer, only a single weight will be mutated during each mutation process. If the value you enter here is half of the result of input layer * output layer, two weights in the weight matrix will be mutated.
        
        genome_fitness (float): Fitness value of genome

    Returns:
        tuple: A tuple containing:
            - mutated_weight (numpy.ndarray): The weight matrix after mutation.
            - mutated_activations (list): The list of activation functions after mutation.

    Notes:
        - Weight mutation:
            - Each weight has a chance defined by `weight_mutate_prob` to be altered by adding a random value 
              within the range of [0, 1].
        - Activation mutation:
            - If `activation_mutate_prob` is triggered, one or more activation functions can be added, removed, 
              or replaced based on their respective probabilities (`activation_add_prob`, `activation_delete_prob`, 
              `activation_change_prob`).
        - The mutation probabilities should be chosen carefully to balance exploration and exploitation during 
          the optimization process.
   """

   if isinstance(activations, str): activations = [activations]

   weight_mutate_prob = 1 - weight_mutate_prob # if prob 0.8 (%80) then 1 - 0.8. Because 0-1 random number probably greater than 0.2
   potential_weight_mutation = random.uniform(0, 1)

   if potential_weight_mutation > weight_mutate_prob:

      start = 0
      row_end = weight.shape[0]
      col_end = weight.shape[1]

      threshold = threshold * genome_fitness
      performance_control = 0
      new_threshold = threshold
      
      while True:
         
         selected_row = int(random.uniform(start, row_end))
         selected_col = int(random.uniform(start, col_end))

         weight[selected_row, selected_col] = random.uniform(-1, 1)

         if row_end * col_end > new_threshold:
            new_threshold += threshold
            performance_control += 1
            pass
         
         else:
            break

         if performance_control == row_end * col_end:
            break
   
   activation_mutate_prob = 1 - activation_mutate_prob
   potential_activation_mutation = random.uniform(0, 1)

   if potential_activation_mutation > activation_mutate_prob:
      
      except_this = ['spiral', 'circular']
      all_acts = [item for item in all_activations() if item not in except_this] # SPIRAL AND CIRCULAR ACTIVATION DISCARDED
      
      activation_add_prob = 1 - activation_add_prob
      activation_delete_prob = 1 - activation_delete_prob
      activation_change_prob = 1 - activation_change_prob

      potential_activation_add_prob = random.uniform(0, 1)
      potential_activation_delete_prob = random.uniform(0, 1)
      potential_activation_change_prob = random.uniform(0, 1)

      if potential_activation_add_prob > activation_add_prob:

         try:
            
            random_index_all_act = int(random.uniform(0, len(all_acts)-1))
            activations.append(all_acts[random_index_all_act])

         except:

            activation = activations
            activations = []

            activations.append(activation)
            activations.append(all_acts[int(random.uniform(0, len(all_acts)-1))])
            
      if potential_activation_delete_prob > activation_delete_prob and len(activations) > 1:
         
         random_index = random.randint(0, len(activations) - 1)
         activations.pop(random_index)
         
         
      if potential_activation_change_prob > activation_change_prob:
         
         random_index_all_act = int(random.uniform(0, len(all_acts)-1))
         random_index_genom_act = int(random.uniform(0, len(activations)-1))

         activations[random_index_genom_act] = all_acts[random_index_all_act]

   return weight, activations

def second_parent_selection(good_weights, bad_weights, good_activations, bad_activations, bad_genomes_selection_prob):
   
   selection_prob = random.uniform(0, 1)
   random_index = int(random.uniform(0, len(good_weights) - 1))

   if selection_prob > bad_genomes_selection_prob:
      second_selected_W = good_weights[random_index]
      second_selected_act = good_activations[random_index]

   else:
      second_selected_W = bad_weights[random_index]
      second_selected_act = bad_activations[random_index]

   return second_selected_W, second_selected_act, random_index
      
def dominant_parent_selection(bad_genomes_selection_prob):

   selection_prob = random.uniform(0, 1)

   if selection_prob > bad_genomes_selection_prob: decision = 'first_parent'
   else: decision = 'second_parent'

   return decision