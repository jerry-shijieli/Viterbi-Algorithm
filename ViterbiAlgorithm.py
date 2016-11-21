#Viterbi algorithm to calculate the most probable path in a 3-state Hidden Markov Model.

#Reference: [Viterbi Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm)

import numpy as np
import csv

# Observations
observations = [1,3,2,4,4,5] # O_i
# Emission matrix
emissions = [
    [0.4, 0, 0],
    [0.2, 0.2, 0.2],
    [0.2, 0.6, 0.1],
    [0.1, 0.1, 0.4],
    [0.1, 0.1, 0.3]
] # emission[X][Z] = P(X|Z)
# State transition matrix
transitions = [
    [0.6, 0.4, 0],
    [0, 0.2, 0.8],
    [0, 0, 0.3]
] # transitions[Zi][Zj] = P(Zi -> Zj)
# initial state
initState = [1.0, 0, 0] # assume at Z1

def viterbi(observations, emissions, transitions, initState):
    dimZ = len(transitions) # dimension of states space
    dimX = len(emissions) # dimension of observations space
    numO = len(observations) # path length of observation sequence
    T1 = np.matrix([[0.0 for _ in range(numO)] * dimZ], dtype=np.float64).reshape([dimZ, numO]) # matrix of observation probability 
    T2 = np.matrix([[0.0 for _ in range(numO)] * dimZ], dtype=np.int32).reshape([dimZ, numO]) # matrix of most likely previous states 
    # initialization time t=0
    transitions = np.matrix(transitions, dtype=np.float64)
    emissions = np.matrix(emissions, dtype=np.float64)
    for i in range(len(initState)):
        T1[i, 0] = initState[i] * emissions[observations[0]-1, i] 
        T2[i, 0] = np.argmax(initState)+1
    # Dynamically compute the most prob path
    for s in xrange(1, numO):
        obs = observations[s]
        for z in xrange(dimZ):
            prevO = np.asarray(T1.transpose()[s-1], dtype=np.float64)
            transZ = np.asarray(transitions.transpose()[z], dtype=np.float64)
            T1[z, s] = emissions[obs-1, z] * np.max(prevO * transZ)
            T2[z, s] = np.argmax(prevO * transZ)+1;
    # Backtrack the most prob states
    states = [-1 for _ in range(numO)]
    z = np.argmax(np.asarray(T1.transpose()[numO-1], dtype=np.float64))+1
    states[numO-1] = z
    for s in range(numO-1, 0, -1):
        z = T2[z-1, s]
        states[s-1] = z
        
    return T1, T2, states

def saveResults(prob, trans, path):
    with open("P2_solution.csv", 'wb') as csvfile:
        cwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        cwriter.writerow(["Probability on each Observation with State Z (row) vs Observations (column): \n"])
        header = ['', 'O1', 'O3', 'O2', 'O4', 'O4', 'O5']
        cwriter.writerow(header)
        index = 0
        for term in prob.tolist():
            index += 1
            term = ["Z"+str(index)] + term
            cwriter.writerow(term)
        cwriter.writerow('\n')
        cwriter.writerow(["Transition Z(t-1) -> Z(t) with Previous State Z (row) vs Time (column): \n"])
        cwriter.writerow(header)
        index = 0
        for term in trans.tolist():
            index += 1
            term = ["Z"+str(index)] + term
            cwriter.writerow(term)
        cwriter.writerow('\n')
        cwriter.writerow(["Most probable path through the states for the observed sequence: "])
        result = ""
        for i in range(len(path)):
            if i==len(path)-1 :
                result += 'Z'+str(path[i])
            else:
                result += 'Z'+str(path[i])+" -> "
        cwriter.writerow([result])

def main():
    Probability, Transition, Path = viterbi(observations, emissions, transitions, initState)
    print "Probability on each Observation with State Z (row) vs Observations (column): "
    print Probability
    print "Transition Z(t-1) -> Z(t) with Previous State Z (row) vs Time (column): "
    print Transition
    print "Most probable path through the states for the observed sequence: "
    print Path
    saveResults(Probability, Transition, Path)

if __name__ == "__main__":
    main()