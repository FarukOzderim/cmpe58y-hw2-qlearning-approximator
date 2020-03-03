import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env.reset()
allData=[]
allData.append(np.zeros(1000))
allData.append(np.zeros(1000))
allData.append(np.zeros(1000))
allData.append(np.zeros(1000))

maxData=[]
minData=[]
for _ in range(4):
	maxData.append([])
	minData.append([])

for j in range(10):
	env.reset()
	for tick in range(500):
	# you don’t have to render. it’s just for visualization.
		#env.render()
		# take a random action
		observation, reward, done, info = env.step(env.action_space.sample())
		for i in range(4):
			allData[i][tick]=(observation[i])
	for i in range(4):
		maxData[i].append(allData[i].max())
		minData[i].append(allData[i].min())
		print("Max and min of:",i, (allData[i].max()), (allData[i].min()))

for i in range(4):
	print("$$$Max and min of:",i, (max(maxData[i])), min(minData[i]))

env.close()

print("plotting 1")
plt.plot(allData[0])
plt.show()


print("plotting 2")
plt.plot(allData[1])
plt.show()

print("plotting 3")
plt.plot(allData[2])
plt.show()

print("plotting 4")
plt.plot(allData[3])
plt.show()