import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
plotData=np.zeros(1000)
#Use the pole angle and pole angle speed data
#Position and velocity of the cart is a little side objective, and can be solved by solving pole angle&angle velocity

thetas=np.array([[np.random.random()*10,np.random.random()*10,np.random.random()*10,np.random.random()*10,np.random.random()*10],[np.random.random()*10,np.random.random()*10,np.random.random()*10,np.random.random()*10,np.random.random()*10]])


#ALPHA=0.01
#epsilon=0.01
gama=0.99

#Epsilon starts with 1 and drops to 0.1 with time
def chooseEpsilon(time):
	return 0.99 ** time
#Alpha starts with 1 and drops to 0.1 with time
def chooseAlpha(time):
    return 0.1
        # return max(0.1,min(1.0,1.0- math.log10((1+time)/10)))

#Normalize Pole Angle to [-pi,+pi]
def discretizePoleAngle(angle):
	angle= angle%np.radians(360)
	angle=angle-np.radians(180)
	return angle

#Choose random action with probabilty epsilon
def chooseAction(epsilon,oldObservation):
	if np.random.random()<epsilon:
		return int(np.random.random()*2)
	else:
		return np.argmax(outputOfModel(oldObservation))

def activationFunction(x):
	return 1/(1+math.exp(-1*x))

#Update Model
def updateModel(oldObservation,newObservation,action,reward,alpha):
	thetasUpdate=np.array([0.0,0.0,0.0,0.0,0.0])
	oldOutput=outputOfModel(oldObservation)
	newOutput=outputOfModel(newObservation)
	for j in range(4):
		thetasUpdate[j]+=-1*alpha*(oldObservation[j])*(oldOutput[action]-(reward+gama*np.max(newOutput)))
	thetasUpdate[4]+=-1*alpha*(oldOutput[action]-(reward+gama*np.max(newOutput)))
	#print(thetasUpdate)
	for j in range(5):
		thetas[action][j]+=thetasUpdate[j] 
		if thetas[action][j]>0:
			thetas[action][j]=min(100.0,thetas[action][j])
		else:
			thetas[action][j]=max(-100.0,thetas[action][j])		
	#return thetasUpdate
#Returns output of the model
def outputOfModel(state):
	outputs=np.zeros(2)
	for i in range(4):
		outputs[0]+=(state[i])*thetas[0][i]
		outputs[1]+=(state[i])*thetas[1][i]
	outputs[0]+=thetas[0][4]
	outputs[1]+=thetas[1][4]
	
#	print(outputs)
	
	#outputs[0]=activationFunction(outputs[0])*10
	#outputs[1]=activationFunction(outputs[1])*10

	return outputs

def convertState(state):
	'''	
	state[0]=state[0]/30
	state[1]=state[1]/20
	state[2]=state[2]/np.radians(180)
	state[3]=state[3]/20
	'''
	return state

accumulatedReward=0
#First 500 rounds, mainly for training
for time in range(500):
	roundSurvival=0
	alpha=chooseAlpha(time)
	epsilon=chooseEpsilon(time)
	oldObservation=env.reset()
	oldObservation[2]=discretizePoleAngle(oldObservation[2])
	oldObservation=convertState(oldObservation)
	#updateTheta1=np.zeros(5)
	#updateTheta2=np.zeros(5)

#	print(thetas)
	for survival in range(500):		
		#env.render()
		action=chooseAction(epsilon,oldObservation)
		newObservation, reward, done, info = env.step(action)
		accumulatedReward+=reward
		roundSurvival+=reward
		'''if done:
			print(time, "Survival tick is:",survival, epsilon)
			#accumulatedReward+=survival
			break'''
		
		newObservation[2]=discretizePoleAngle(newObservation[2])
		newObservation=convertState(newObservation)
		updateModel(oldObservation,newObservation,action,reward,alpha)
		'''if action==1:
			updateTheta1+=updateModel(oldObservation,newObservation,action,reward,alpha)
		else:
			updateTheta2+=updateModel(oldObservation,newObservation,action,reward,alpha)
			'''
		#updateTheta1+=x[0]
		#updateTheta2+=x[1]
		oldObservation=newObservation
	#thetas[0]+=updateTheta1
	#thetas[1]+=updateTheta2

	plotData[time]=roundSurvival
	print(time, "Round reward and epsilon is:",roundSurvival, epsilon)

print(thetas)
average1=accumulatedReward/500
print("Average rewards in first 500 rounds is:",average1)

env.close()

print(thetas)