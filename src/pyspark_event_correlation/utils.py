def fixing_steps(k, p, steps, file, algorithm="shewhart"):
	# k 1...7 -> number of events
	# p 0
	# step 1..3
	# algorithm cusum shewhart
	k = int(k)
	p = float(p)
	steps = int(steps)
	graph = {}
	# freq = {}
	res = open(file, "w")
	res.write("STEPWISE CORRELATION,\tFIXED "+str(k)+",\tPROBABILITY GREATER THAN "+str(p)+",\t")
	csvFile = "eventVectors/" + algorithm + "/"
	if k == 0:
		csvFile = csvFile + algorithm + "EventVector.csv"
		res.write(algorithm+" ALGORITHM\n")
	else:
		csvFile = csvFile + algorithm + "RandomKevents" + str(k) + ".csv"
	events = pd.read_csv(csvFile, header=None, squeeze=True)

	events = np.array(events)
	train = events[:1000]
	test = events[1000:]

	#FIRST VECTOR
	event = ''.join(map(str,events[0])) # '00000000000000000000000000000'
	graph[event] = {}
	# freq[event] = 1

	#TRAINING DATA
	t = 0 # size of training set - 1
	for i in train[1:]:
		prevState = event
		event = ''.join(map(str,i))
		updateGraph(prevState, event, graph)
		# updateFreq(event, freq)
		t += 1



	# exei ftiaxtei to graph
	# print(event)
	#TESTING DATA
	f = open("predictions.csv", "w")
	#FIRST PREDICTION


	predictions = 0
	###################################################
	# if graph[event]:
	# 	pred = max(graph[event], key=graph[event].get)
	# 	# preddar.append(pred)
	# 	predictions+=1
	# else:
	# 	pred = None
	# entry = str(event)+" "+str(pred)+" "
	# if pred == None:
	# 	entry += "0\n"
	# else:
	# 	entry += str(graph[event][pred]/(t-1)) + "\n"
	# f.write(entry)
	###################################################
	w = 0
	preddar = []
	for i in test:

		#NEW EVENT
		prevState = event
		event = ''.join(map(str,i))

		#UPDATE GRAPH
		updateGraph(prevState, event, graph)
		# updateFreq(event, freq)
		# print(graph[event])

		#CHECK IF ABLE TO MAKE PREDICTION


		if graph[event]:
			#random in case of a tie
			maxi = max(graph[event].values())

			if (maxi/t) >= p:
				pred = random.choice([k for (k,v) in graph[event].items() if v==maxi])
				predictions+=1
				preddar.append(pred)
				w +=1
				if len(preddar) == 4:
					# print(graph[event].items())
					# print(i)
					pass
					# print(maxi)
			else:
				pred = None
		else:
			pred = None


		entry = str(event)+" "+str(pred)+" "
		if pred == None:
			entry += "0\n"
		else:
			entry += str(maxi/t) + "\n"
		f.write(entry)
		t += 1

	f.close()


	results = pd.read_csv("predictions.csv", delimiter=" ")
	results = np.array(results)
	res.write("PROBABILITY / TRUE/FALSE\n")
	exact = 0
	for i in range(0,len(results)):
		pred = results[i,1]
		prob = results[i,2]
		flag = 0
		for j in range(1,steps+1):
			if i+j < len(results):
				event = results[i+j,0]
				# CHECK IF PREDICTION WAS CORRECT
				if pred != "None":
					s1 = int(pred, base=2)
					s2 = int(event, base=2)
					bitand = '{0:029b}'.format(s1 & s2)

					if bitand == pred and s1 == s2:
					# if s1 == s2:
						exact+=1
						flag = 1
						break


		res.write(str(prob)+" "+str(flag)+"\n")

	precision = (exact/predictions)*100
	recall = (exact/len(test))*100

	print(exact)
	print("Precision is: " + str(precision) + "%")
	res.write("Precision is: " + str(precision)+"%\n")
	print("Recall is "+ str(recall) + "%")
	res.write("Recall is: " + str(recall)+"%\n")