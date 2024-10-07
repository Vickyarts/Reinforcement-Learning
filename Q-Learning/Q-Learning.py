import numpy as np 
import random 




states = np.array(range(16))
actions = ['up', 'down', 'right', 'left']

Q_table = np.zeros((len(states), len(actions))) 


states = states.reshape(4,4)


goal = 15
penalty = -1


alpha = 0.1                                # Factor that alters the Temporal Difference added to the Q-Value.
gamma = 0.9                                # Discount factor in Bellman eqn.
epsilon = 0.1                              # 10% The probability of making a random move instead of making the move with best Q-Value.
episodes = 1000                            # Like epochs.




def next_state(current_state, action):     # Function to find the next state using the action from current state
    row = int(current_state / 4)
    col = current_state % 4
    
    if action == 'up': 
        if row != 0 : 
            row -= 1
            
    elif action == 'down': 
        if row !=3: 
            row += 1
        
    elif action == 'right': 
        if col != 3: 
            col += 1
        
    elif action == 'left': 
        if col != 0: 
            col -= 1
        
    return states[row, col]




for episode in range(episodes): 
    state = random.randint(0, 15)  # Choosing new random state for each episode to explore through the Maze.
    
    while state != goal:                                        # Go through the maze until we reach the goal.
        explore_probability = random.uniform(0, 1)              # Adding Randomness. To choose make random move or make the best move.
        if explore_probability < epsilon: 
            action_index = random.choice(range(len(actions)))   # Choosing random action.
        else: 
            action_index = np.argmax(Q_table[state])            # Choosing best action from that state.
            
        nx_state = next_state(state, actions[action_index])     # Getting next state from the function. 
        
        if nx_state == 15:                                      # reward = 100 if it reached the goal.
            reward = 100
        else:                                                   # reward =- 1 if not (living penalty).
            reward = -1
        
        Q_table[state, action_index] = Q_table[state, action_index] + alpha * (reward + (
            gamma*np.max(Q_table[nx_state]) - Q_table[state, action_index]
            ))                                                              # Calculating the Q-Value for the current state and assigning it to the table.
        
        
        state = nx_state                                                    # Moving to the next state.



optimal_route = []
for state in states.reshape(-1):                                            # Getting the optimal move at a state. (Model already trained)
    optimal_route.append(actions[np.argmax(Q_table[state])])
optimal_route = np.array(optimal_route)


print(optimal_route.reshape(4,4))                                          # Visualizing the optimal moves in (4x4) as the Maze(4x4).
