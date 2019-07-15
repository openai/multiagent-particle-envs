import gym
env = gym.make("MountainCar-v0")

done=False
state=env.reset()
while not done:
    if state[1]<=0:
        state, reward, done,info = env.step(0)
    else:
        state, reward, done,info = env.step(2)
    env.render()
    
env.close()