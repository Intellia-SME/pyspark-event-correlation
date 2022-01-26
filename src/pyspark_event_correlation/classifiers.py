import time
from math import exp
import pandas as pd
import numpy as np

from .utils import fixing_steps
from pyspark.ml.classification import LogisticRegression

def powerset(lst):
    # the power set of the empty set has one element, the empty set
	result = lst
	res = list(map(list, result))
	return res

# k = ageing factor, i = how many steps is the prob active
def linearAgeing(i):
	age = 1 - 0.05*i
	return age

def exponentialAgeing(i):
	k = 0.05
	age = exp(-k*i)
	return age


def updateFreq(event, freq):
	if event not in freq:
		freq[event] = 0
	freq[event] += 1


def updateGraph(prevState, event, graph):
	if event not in graph:
		graph[event] = {}

	if event not in graph[prevState]:
		graph[prevState][event] = 0
	graph[prevState][event] += 1

def sliding(k, w, p, steps, ageingFunction, file, algorithm = "shewhart"):
	# k 1...7
	# w 3...7
	# steps 1...3
	# ageing linear/exponential

	k = int(k)
	w = int(w)
	p = float(p)
	steps = int(steps)
	columns = pd.read_csv("detectionAlgorithms/DATASET_CMA_CGM_NERVAL_5min.csv",  nrows = 0)
	columns = tuple(columns.iloc[:,:-1])
	streams = bitset('streams', columns)

	# GET k FIXED EVENT VECTORS
	res = open(file, "w")
	res.write("SLIDING WINDOW,\tWINDOW LENGTH "+str(w)+",\tFIXED "+str(k)+",\tPROBABILITY GREATER THAN "+str(p)+",\t"+ algorithm+" ALGORITHM\n,\t"+ageingFunction+" ageing\t")
	csvFile = "eventVectors/" + algorithm + "/"
	if k == 0:
		csvFile = csvFile + algorithm + "EventVector.csv"
	else:
		csvFile = csvFile + algorithm + "RandomKevents" + str(k) + ".csv"
	events = pd.read_csv(csvFile, header=None, squeeze=True)
	events = np.array(events)

	if ageingFunction == "linear":
		ageingFun = linearAgeing
	else:
		ageingFun = exponentialAgeing

	# CONSTANTS
	prevPSets = []
	prevKeys = []
	tim = 0
	# PREPARE UP TO w-1 EVENT VECTORS
	w = 3
	for i in range(0,w-1):
		selected = streams.frombits(events[i])
		pSet = []

		pSet = list(powerset(list(selected.members())))
		if len(pSet) != 1:
			pSet = pSet[1:]
		# get the powerset for each event vector and the key for predictions
		prevPSets.append(pSet)
		prevKeys.append(events[i])

	prediction = (None, 0)
	numOfPreds = 0

	f = open("predictions"+str(get_ident())+".csv", "w")

	entry = str(prevKeys[w-2])+" 0 0\n"
	f.write(entry)
	ageing = []
	for event in events[w-1:]:
		prediction = (None, 0)

		# enter the current event to the table
		selected=[]
		selected = streams.frombits(event)
		currPS = list(powerset(list(selected.members())))
		if len(currPS) != 1:
			currPS = currPS[1:]
		prevPSets.append(currPS)
		prevKeys.append(event)



		# get all sets in the powersets that have appeared in the time window
		powerList = []
		# from list to tuple
		# prevPSets -> [[], [' INCLINOMETERYZC']]
		# powerList -> [(), (' INCLINOMETERYZC',)]
		powerList = list(set([tuple(item) for sublist in prevPSets for item in sublist]))


		start = time.time()
		# iterate for each couple of sets within the powerset list to get the one that has the maximum probability
		for k in range(0,len(powerList)):

			# the given set
			item = list(powerList[k])

			# start from the first encounter of the given set in the table

			i = 0
			while item not in prevPSets[i]:
				i += 1
			for l in range(k,len(powerList)):

				# the set that its probability is being calculated currently
				psItem = list(powerList[l])

				# from there onward, calculate the probability of the set
				psItemCount = idealCount = 0
				for j in range(i, w):
					if item in prevPSets[j]:
						idealCount += (w-j)
						for m in range(j,w):
							if psItem in prevPSets[m]:
								psItemCount += 1
				prob = psItemCount/idealCount

				# get the maximum probability of all combinations
				if prob > prediction[1]: # and prob >= p:
					prediction = (psItem, prob)
		# print(prediction)

		end = time.time()


		tim += end-start

		if prediction[0] != None:
			prediction = (streams(prediction[0]).bits(), prediction[1])
			numOfPreds += 1

		#for all previous sets of predictions, check which one is the best
		if prediction[0] != None:

			if ageing != [] and ageing[0][2] == 20:
				ageing.pop(0)

		best = (prediction[0], prediction[1])
		flag = 0
		temp = (0,0)
		for r in range(0,len(ageing)):
			ag = ageing[r]
			ag = (ag[0], ag[1], ag[2]+1)
			# if the prediction is included in ???? poio mesa se poio?
			if prediction[0] == None or ag[0] == None:
				continue
			binPred = int(prediction[0], base=2)
			binAg = int(ag[0], base=2)
			if '{0:029b}'.format(binPred & binAg) == ag[0]:
				flag = r
				if ag[1]*ageingFun(ag[2]) > prediction[1]:
					# change both value and prediction?????
					prediction = (ag[0], ag[1]*ageingFun(ag[2]))
				# ageing.pop(r)
				temp = (prediction[0], prediction[1], 0)
			if ag[1]*ageingFun(ag[2]) > best[1]:
				best = (ag[0], ag[1]*ageingFun(ag[2]))

		# append first ageing values
		if ageing == [] or not flag:
			ageing.append((prediction[0], prediction[1], 0))

		if temp != (0,0):
			ageing.pop(flag)
			prediction = temp


		entry = str(event)+" "+str(prediction[0])+" "+str(prediction[1])+"\n"
		f.write(entry)
		# remove the oldest entry
		prevKeys.pop(0)
		prevPSets.pop(0)


	f.close()

	# print(tim)


	results = pd.read_csv("predictions"+str(get_ident())+".csv", delimiter=" ")
	results = np.array(results)
	res.write("PROBABILITY / TRUE/FALSE\n")

	exact = 0
	recallExact = 0
	for i in range(0,len(results)):
		pred = results[i,1]
		prob = results[i,2]
		flag = 0
		mutex.acquire()
		for j in range(1,steps+1):
			if i+j < len(results):
				event = results[i+j,0]

				# check if previous prediction was correct
				if pred != "None":
					binPred = int(pred, base=2)
					binEvent = int(event, base=2)
					bitand = '{0:029b}'.format(binPred & binEvent)
					if bitand == pred:
						exact += 1
						flag = 1
						if int(bitand, base=2) <= binEvent:
							recallExact += 1
						break
		mutex.release()

		res.write(str(prob)+" "+str(flag)+"\n")
	print(exact)

	if numOfPreds == 0:
		precision = 0
	else:
		precision = (exact/numOfPreds)*100
	recall = (recallExact/len(events[w-1:]))*100
	# print("Precision is: " + str(precision) + "%")
	res.write("Precision is: " + str(precision) + "%\n")
	# print("Recall is "+ str(recall) + "%")
	res.write("Recall is: " + str(recall) + "%\n")
	os.remove("predictions"+str(get_ident())+".csv")

class EventCorrelationClassifier(LogisticRegression):
    pass
