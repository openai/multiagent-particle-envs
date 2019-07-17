import gym
import numpy 

env = gym.make("MountainCar-v0")

learningrate = 0.7
discount = 0.90
#initiallize the Q table [40,40,3] with random values.  The meaning of the q table is the q value of a set of [state of positions,state of velocity, action you take]. 
#Note that the game is continous but the states of our q table are discrete(since we can only deal with finite states), So I also need a getstate function to turn the continous states into deiscrete states.
#all q values are initialized between -2 and 0 because the reward is always -1 in the mountaincar game.
q_table = numpy.random.uniform(-2, 0, [40,40,3])


def getstate(state):
    discrete_state = (state - env.observation_space.low)/((env.observation_space.high-env.observation_space.low)/[40,40])
    return tuple(discrete_state.astype(numpy.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


for episode in range(2700):
    currentstate = getstate(env.reset())
    done = False
    #render every 300 episodes to save time.
    if episode % 300 == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        action = numpy.argmax(q_table[currentstate])
        new_state, reward, done,info = env.step(action)
        #nextstate is the discrete mapping from the new state to the q table
        nextstate = getstate(new_state)

        if render:
            env.render()

        # Update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            maxnextq = numpy.max(q_table[nextstate])
            # Current Q value (for current state and performed action)
            current_q = q_table[currentstate + (action,)]
            # the qlearning function
            new_q = (1 - learningrate) * current_q + learningrate * (reward + discount * maxnextq)
            # Update Q table with new Q value
            q_table[currentstate + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= 0.5:
            print("We make it!")
            print(episode)
            q_table[currentstate + (action,)] = 0
            

        currentstate = nextstate

   


env.close()