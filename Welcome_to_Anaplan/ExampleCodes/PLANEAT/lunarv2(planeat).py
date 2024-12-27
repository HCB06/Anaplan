import gym
from anaplan import planeat
import numpy as np

"""
pip install gym
pip install box2d-py
"""

# Ortam oluşturma
env = gym.make('LunarLander-v2')
state = env.reset(seed=0)
state = np.array(state[0])

# Genomlar ve jenerasyon sayısı
genome_weights, genome_activations = planeat.define_genomes(input_shape=8, output_shape=4, population_size=300)
generation = 0
rewards = [0] * 300

reward_sum = 0

while True:
    for i in range(300):
        while True:
            
            # Aksiyon hesaplama
            output = planeat.evaluate(
                x_population=np.array(state),
                weights=genome_weights[i],
                rl_mode=True,
                activation_potentiations=genome_activations[i]
            )
            action = np.argmax(output)
            state, reward, done, truncated, _ = env.step(action)
            
            reward_sum += reward
            
            if done or truncated:
                state = env.reset(seed=0)
                state = np.array(state[0])
                rewards[i] = reward_sum
                reward_sum = 0
                
                break
            
        if i > 298:
            env = gym.make('LunarLander-v2', render_mode='human')
            state = env.reset(seed=0)
            state = np.array(state[0])
        else:
            env.close()
            env = gym.make('LunarLander-v2')
            state = env.reset(seed=0)
            state = np.array(state[0])

    # Jenerasyon ve genom güncellemesi
    generation += 1
    
    genome_weights, genome_activations = planeat.learner(
        genome_weights,
        genome_activations,
        generation,
        strategy='cross_over',
        policy='normal_selective',
        y_reward=np.array(rewards),
        mutations=True,
        show_info=True
    )
    
    if generation == 20:
        
        break
