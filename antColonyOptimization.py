
from operator import ge
import random

INT_MAX = 1e15
METRIC_FLAG = 1 # 0- OperationalCost aware,  1- Delay aware

class Ant :
	trail = []
	trailOperationalCost = 0.0
	trailDelay = 0.0

	# O(1)
	def __init__(self, numberOfNodes) :
		self.numberOfNodes = numberOfNodes

	# O(1)
	def visitNode(self, node, vnf, operationalCost, delay) :
		self.trail.append([node, vnf])
		self.trailOperationalCost += operationalCost
		self.trailDelay += delay

	# O(1)
	def getTrailLength(self) : 
		return len(self.trail)

	# O(1)
	def getTrail(self) : 
		return self.trail
	
	# O(1)
	def getTrailOperationalCost(self) :
		return self.trailOperationalCost

	# O(1)
	def getTrailDelay(self) :
		return self.trailDelay

	# O(1)
	def getMetricValues(self) :
		if(METRIC_FLAG == 0) :
			return self.trailOperationalCost, self.trailDelay
		elif(METRIC_FLAG == 1) :
			return self.trailDelay, self.trailOperationalCost

	# O(1)
	def clear(self) :
		self.trail = []
		self.trailOperationalCost = 0.0
		self.trailDelay = 0.0

class AntColonyOptimization : 
	alpha = 0.7
	beta = 0.7
	evaporation = 0.7
	Q = 10.0 # factor so that pheromones value do no get too small
	antFactor = 200
	maxIterations = 10
	maxVnf = 8
	defaultVnf = 0 # packet forwarder index in dictionary

	ants = []
	graph = []
	bestTrail = []
	edgeInBestTrail = {}
	pheromones = []

	bestTrailOperationalCost = INT_MAX + 7
	bestTrailDelay = INT_MAX + 7

	# O(numberOfNodes^2 * vnfNumber) + O(Solve) 
	# = O(maxIteration * 500 * numberOfNodes^4 * vnfNumber)
	# = O(9 * 1e8)
	def __init__(self, graph, start, end, vnfList, vnfCPU, nodeConstant, nodeCPU, thresholdDelay) : 
		self.graphEdgeList = graph
		self.minTrailLength = len(vnfList) + 2
		self.numberOfNodes = len(graph)
		self.maxTrailLength = (self.numberOfNodes)
		self.numberOfAnts = int(self.numberOfNodes * self.antFactor)
		self.thresholdDelay = thresholdDelay

		self.nodeCPU = nodeCPU[:]
		self.vnfList = vnfList
		self.nodeConstant = nodeConstant

		self.vnfCPU = {x: 0 for x in range(self.maxVnf)}

		for x in vnfCPU :
			self.vnfCPU[x] = vnfCPU[x]

		for i in range(self.numberOfNodes) :
			self.graph.append([-1]*(self.numberOfNodes))

		for i in range(self.numberOfNodes) :
			for x in graph[i]:
				self.graph[i][x[0]] = x[1]

		for i in range(self.numberOfNodes) :
			ll = []
			for j in range(self.numberOfNodes) :
				l = []
				for k in range(self.maxVnf) :
					l.append(0.0)
				ll.append(l)
			self.pheromones.append(ll)
				
		
		self.start = start
		self.end = end

		# self.probabilities = [0] * self.numberOfNodes
		for i in range(self.numberOfAnts) :
			self.ants.append(Ant(self.numberOfNodes))

		# print(len(self.ants))
		# print (self.numberOfNodes)
		self.solve()

	# O(1)
	def getEdgeWeight(self, source, target) :
		return self.graph[source][target]        

	# O(1)
	def getOperationalCost(self, targetNode, vnf) :
		if(vnf < 0) :
			return 0.0
		return self.nodeConstant[targetNode] * self.vnfCPU[vnf]

	# O(1)
	def getMetric(self, sourceNode, targetNode, vnf) :
		if(METRIC_FLAG == 0) :
			return self.getOperationalCost(targetNode, vnf)
		elif(METRIC_FLAG == 1) :
			return self.getEdgeWeight(sourceNode, targetNode)

	# O(1)
	def getBestMetricValues(self) :
		if(METRIC_FLAG == 0) :
			return self.bestTrailOperationalCost, self.bestTrailDelay
		elif(METRIC_FLAG == 1) :
			return self.bestTrailDelay, self.bestTrailOperationalCost

	# O(1)
	def updateBestMetric(self, metric1, metric2) :
		if(METRIC_FLAG == 0) :
			self.bestTrailOperationalCost = metric1
			self.bestTrailDelay = metric2
		elif(METRIC_FLAG == 1) :
			self.bestTrailDelay = metric1
			self.bestTrailOperationalCost = metric2

	# O(numberOfNodes^2 * vnfNumber)
	def initPheromones(self) :
		for i in range(self.numberOfNodes) :
			for j in range(self.numberOfNodes) :
				for k in range(self.maxVnf) :
					metric = float(self.getMetric(i, j, k))
					if (metric > 0) : 
						self.pheromones[i][j][k] = self.Q/metric #magnification

	# O(numberOfNodes + 2*numberOfEdges * getMetric()) 
	# = O(numberOfNodes)
	def calculateProbability(self, currentNode, vnf) : 
		probabilities = [0] * (self.numberOfNodes)
		sum = 0.0

		for edge in self.graphEdgeList[currentNode] :
			nextNode = edge[0]
			metric = self.getMetric(currentNode, nextNode, vnf)
			nij = self.Q/metric #magnification
			tij = self.pheromones[currentNode][nextNode][vnf]
			sum += ((tij ** self.alpha) * (nij ** self.beta))

		if(sum == 0) :
			return probabilities

		for edge in self.graphEdgeList[currentNode] :
			nextNode = edge[0]
			metric = self.getMetric(currentNode, nextNode, vnf)
			nij = self.Q/metric #magnification
			tij = self.pheromones[currentNode][nextNode][vnf]
			probabilities[nextNode] = ((tij ** self.alpha) * (nij ** self.beta)) / sum
		
		return probabilities

	# O(calculateProbability() + 2 * numberOfNodes) 
	# = O(numberOfNodes)
	def findNextNode(self, currentNode, vnf) :
		probabilities = self.calculateProbability(currentNode, vnf)
		sum = 0.0
		nextNode = -1
		for i in range( len(probabilities)) :
			probabilities[i] *= 100
			sum += probabilities[i]

		# print(sum)
		if (sum == 0) :
			return -1
		# random number between (0, sum] = [1,sum] - [0, 1) 
		number = random.uniform(1.0,sum) - random.random() 
		prev = 0.0
		for i in range(len(probabilities)) :
			if (prev <= number and number <= prev + probabilities[i]) :
				nextNode = i
				break
			prev += probabilities[i]

		# print(currentNode, "--",probabilities, "--", nextNode)
		#check if it returns a single value or list
		# todo = update random selection --done
		# nextNode = random.choices(range(len(probabilities)), weights=probabilities, k=1)
		return nextNode

	# O(numberOfBestTrails (~ 5-10) * lengthOfBestTrail (< numberOfNodes)) 
	# = O(numberOfNodes^2)
	# O(1)
	# check if a given edge is present in the best trail
	def checkEdgeInShortestTrail(self, src, dst) :
		if((src, dst) in self.edgeInBestTrail) : 
			return True
		return False

	# O(getBestMetricValues() + numberOfNodes^2 * vnfNumber * checkEdgeInShortestTrail()) 
	# = O(numberOfNodes^2 * vnfNumber)
	# only update if the current path is smallest
	def updatePheromones(self) :
		m1, m2 = self.getBestMetricValues()
		dt = self.Q / float(m1) #magnification

		for i in range(self.numberOfNodes) : 
			for j in range(self.numberOfNodes) :
				for k in range(self.maxVnf) :
					weight = float(self.getMetric(i, j, k))
					dtij = dt if self.checkEdgeInShortestTrail(i,j) else 0.0
					tij = self.pheromones[i][j][k]
					if (weight != -1) :
						self.pheromones[i][j][k] = (1.0 - self.evaporation) * tij + dtij

	# O(1)
	# update on pheromone on path taken by ant
	def updateLocalPheromones(self, currentNode, nextNode, vnf) :
		# todo = move local pheromone update to another function -- done 
		metric = float(self.getMetric(currentNode, nextNode, vnf))
		t0 = 0.0
		if (metric > 0) :
			t0 = self.Q / metric #magnification
		tij = self.pheromones[currentNode][nextNode][vnf]
		self.pheromones[currentNode][nextNode][vnf] = ((1.0 - self.evaporation) * tij) + (self.evaporation * t0)

	# O( maxTrailLength * (findNextNode() + updateLocalPheromones() + visitNode()))
	# = O(numberOfNodes^2)
	# find solution for one ant
	def solveAnt(self, ant) :
		currentNode = self.start
		vnfIndex = 0
		CPU = self.nodeCPU[:]
		while(currentNode != self.end):
			vnf = self.defaultVnf
			if(vnfIndex < len(self.vnfList) ) :
				vnf = self.vnfList[vnfIndex]

			nextNode = self.findNextNode(currentNode, vnf)
			if(nextNode == -1):
				ant.visitNode(-1, -1, INT_MAX, INT_MAX)
				break

			if(CPU[nextNode] < self.vnfCPU[vnf]) :
				vnf = self.defaultVnf
			elif(nextNode == self.end):
				vnf = -1
			else :
				CPU[nextNode] -= self.vnfCPU[vnf]
				vnfIndex += 1

			self.updateLocalPheromones(currentNode, nextNode, vnf)
			
			ant.visitNode(nextNode, vnf, self.getOperationalCost(nextNode, vnf), self.getEdgeWeight(currentNode, nextNode))
			currentNode = nextNode
			
			if(ant.getTrailLength() > self.maxTrailLength or ant.getTrailDelay() > self.thresholdDelay) :
				ant.visitNode(-1, -1, INT_MAX, INT_MAX)
				break
	
		if(vnfIndex < len(self.vnfList)) :
			ant.visitNode(-1, -1, INT_MAX, INT_MAX)

	# O(numberOfAnts) = O(numberOfNodes * antFactor(=500))
	# O(500 * numberOfNodes)
	# initialise all ants for each iteration 
	def initAnts(self) :
		for ant in self.ants :
			ant.clear()
			ant.visitNode(self.start, -1, 0.0, 0.0)

	# O(numberOfBestTrails (~ 5-10) * lengthOfBestTrail (< numberOfNodes)) 
	# = O(numberOfNodes^2)
	def uniqueTrail(self, trail, trails):
		for x in trails:
			if(len(x) == len(trail)):
				eq = True
				for i in range(len(x)) :
					if(x[i][0] != trail[i][0] or x[i][1] != trail[i][1]) :
						eq = False
						break
				if(eq):
					return False
		return True

	# O(initAnts() + numberOfAnts * (solveAnt() + updatePheromones())) 
	# = O(500 * numberOfNodes + 500 * numberOfNodes * (numberOfNodes^2 + (numberOfNodes^4 * vnfNumber)) )
	# = O(500 * numberOfNodes^5 * vnfNumber) = O(9 * 1e7)
	# find solution for for all ants for 1 iteration
	def solveIteration(self) :
		self.initAnts()
		bestTrail = []
		metric1 = INT_MAX + 6
		metric2 = INT_MAX + 6
		for ant in self.ants :
			self.solveAnt(ant)
			if (ant.getTrailLength() >= self.minTrailLength):
				antM1, antM2 = ant.getMetricValues()
				antTrail = ant.getTrail()
				# print(antTrail)
				if (antM1 < metric1) :
					bestTrail = antTrail
					# edgeInBestTrail[(1,2)] = 1
					# (1,2) in edgeInBestTrail
					metric1 = antM1
					metric2 = antM2
					self.updatePheromones()
				elif (antM1 == metric1 and antM2 < metric2) :
					bestTrail = antTrail
					metric1 = antM1
					metric2 = antM2
					self.updatePheromones()
		return bestTrail, metric1, metric2

	# todo - self.edgeInbestTrail update, edgeInBestTrail update, edge in trail update
	# O(initPheromones() + maxIteration * (solveIteration() + maxTrailLength)) 
	# = O(numberOfNodes^2 * vnfNumber + maxIteration * 500 * numberOfNodes^4 * vnfNumber) 
	# = O(9 * 1e8)
	def solve(self) :

		self.initPheromones()
		# print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.pheromones]))

		for i in range(self.maxIterations) :
		# for i in range(100) :
			trail, metric1, metric2 = self.solveIteration()
			bestMetric1, bestMetric2 = self.getBestMetricValues()
			# print(trailWeight, trails)
			if(metric1 < bestMetric1) :
				self.bestTrail = trail
				self.updateBestMetric(metric1, metric2)
				length = len(trail)
				for i in range(length - 1) :
					src = trail[i][0]
					dst = trail[i+1][0]
					# print(type(src), type(dst), type((src, dst)))
					self.edgeInBestTrail[(src, dst)] = 1
				
			elif(metric1 == bestMetric1 and metric2 < bestMetric2) :
				self.bestTrail = trail
				self.updateBestMetric(metric1, metric2)
				length = len(trail)
				for i in range(length - 1) :
					src = trail[i][0]
					dst = trail[i+1][0]
					self.edgeInBestTrail[(src, dst)] = 1
	
	# O(1)
	def getBestTrail(self) :
		return self.bestTrail
		
	# O(1)
	def getBestTrailOperationalCost(self) :
		return self.bestTrailOperationalCost
		
	# O(1)
	def getBestTrailDelay(self) :
		return self.bestTrailDelay

def getSFCPath(graph, degree, start, end, minTrailLength):
	antColony = AntColonyOptimization(graph, start, end, minTrailLength)

	trails = antColony.getBestTrail()

	best_totol_degree = INT_MAX
	ans = trails[0]

	for trail in trails :
		total_degree = sum([degree[x] for x in trail])
		# print(trail)

		if total_degree < best_totol_degree:
			ans = trail
			best_totol_degree = total_degree
	
	print("SFC_1: Path = {0} and Weight = {1}".format(ans, antColony.getBestTrailOperationalCost()))
	return ans

# O(40 * 9 * 1e8)
if __name__ == "__main__":
	file = open("graph.txt", "r")
	content = str(file.read()).split('\n')
	file.close()

	start = int(content[0])
	end = int(content[1])
	edges = int(content[2])
	
	graph = {}

	for i in range(start, end+1):
		graph[i] = []

	for i in range(3, 3+edges):
		line = content[i].split(' ')
		src, dst, wgt = int(line[0]), int(line[1]), float(line[2])
		graph[src].append([dst, wgt])
		
		if(src != start and dst != end):
			graph[dst].append([src, wgt])
	
	# print(graph)


	vnfMapping = {"FW" : 1, "DPI" : 2, "LB1" : 3, "PF": 4, "LB2": 5, "ENC": 6, "DEC": 7}
	vnfCPU = {0 : 0.0001, 1 : 14.32 , 2 : 15.25, 3 : 13.87, 4 : 13.27, 5 : 13.67, 6 : 16.22, 7 : 16.41}

	# nodeConstant = [10.0]*len(graph)
	# nodeConstant = [[9, 9, 7, 9, 8], [9, 10, 7, 7, 5], [9, 8, 7, 6, 9], [5, 7, 6, 5, 9], [9, 9, 5, 7, 8], [5, 10, 10, 9, 6], [8, 7, 10, 9, 7], [5, 10, 6, 5, 7], [7, 10, 9, 10, 6], [7, 9, 9, 9, 9], [6, 5, 5, 9, 5], [10, 9, 6, 9, 8], [6, 7, 9, 10, 5], [5, 6, 5, 9, 8]]
	nodeConstant = [5, 5, 10, 7, 6, 8, 8, 10, 6, 7, 8, 10, 7, 8, 7, 5, 6, 9, 10, 10, 5, 7, 8, 9, 9, 7, 6, 10, 10, 5, 6, 6, 10, 10, 8, 9, 10, 9, 10, 10, 5, 8, 5, 5, 6, 8, 7, 8]

	nodeCPU = [180.0]*len(graph)

	totalOP = 0.0
	totalCPU = 0.0

	file = open("SFC_request.txt", "r")
	content = str(file.read()).split('\n')
	file.close()

	max_delay = 0.0
	
	for no, line in enumerate(content):
		s = ""
		line = line.split(' ')[1:-1:1]
		vnfList = [vnfMapping[x] for x in line]

		thresholdDelay = 50.0

		# print('====debug====')
		antColony = AntColonyOptimization(graph, start, end, vnfList, vnfCPU, nodeConstant, nodeCPU, thresholdDelay)
		# print('====debug====')

		max_delay = max(max_delay, antColony.getBestTrailDelay())

		s += "========== {0} ===========\n".format(no)
		s += "SFC Request = {0}\n".format(vnfList)
		s += "BestTrailOperational = {0}\n".format(antColony.getBestTrailOperationalCost())
		s += "BestTrail = {0}\n".format(antColony.getBestTrail())
		s += "BestTrailDelay = {0}\n".format(antColony.getBestTrailDelay())

		trail = antColony.getBestTrail()
		totalOP += antColony.getBestTrailOperationalCost()

		totalCPU += sum([vnfCPU[x] for x in vnfList])

		for [n, v] in trail:
			if(v > 0):
				nodeCPU[n] -= vnfCPU[v]
		
		s += "nodeCPU = {0}\n".format(nodeCPU)
		s += "#SFC : {0}   TotalOpreationalCost = {1}\n".format((no+1), totalOP)
		s += "#SFC : {0}   TotalCPU_Utilization = {1}\n".format((no+1), totalCPU/(46*180))
		s += "#SFC : {0}   Max_Delay = {1}\n\n".format((no+1), max_delay)


		print(s)
		file = open("output.txt", "a")
		file.write(s)
		file.close()