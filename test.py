import numpy as np
from random import choice
import random
legalPlays = ['U','D','L','R']
N = 4
swapZeroIdxs_dict = dict()
swapZeroIdxs = np.zeros((N**2,len(legalPlays)),dtype=np.int)
for moveIdx,move in enumerate(legalPlays):
    for i in range(N):
        for j in range(N):
            zIdx = np.ravel_multi_index((i,j),(N,N))
            if zIdx not in swapZeroIdxs_dict:
                swapZeroIdxs_dict[zIdx] = []

            state = np.ones((N,N),dtype=np.int)
            state[i,j] = 0
            
            if move == 'U':
                isEligible = i < (N-1)
            elif move == 'D':
                isEligible = i > 0
            elif move == 'L':
                isEligible = j < (N-1)
            elif move == 'R':
                isEligible = j > 0

            if isEligible:
                if move == 'U':
                    swap_i = i+1
                    swap_j = j
                elif move == 'D':
                    swap_i = i-1
                    swap_j = j
                elif move == 'L':
                    swap_i = i
                    swap_j = j+1
                elif move == 'R':
                    swap_i = i
                    swap_j = j-1

                swapZeroIdxs[zIdx,moveIdx] = np.ravel_multi_index((swap_i,swap_j),(N,N))
                swapZeroIdxs_dict[zIdx].append((moveIdx,np.ravel_multi_index((swap_i,swap_j),(N,N))))
            else:
                swapZeroIdxs[zIdx,moveIdx] = zIdx
    
print(swapZeroIdxs)
print(swapZeroIdxs_dict)
solvedState = np.concatenate((np.arange(1,16),[0]))

def next_state(states_input,move):
    move = move.upper()
    states = states_input.copy()
    if len(states.shape) == 1:
        states = np.array([states])

    n_all,zIdxs = np.where(states == 0)

    moveIdx = np.where(["U", "D", "L", "R"] == np.array(move))[0][0]

    stateIdxs = np.arange(0,states.shape[0])
    swapZIdxs = swapZeroIdxs[zIdxs,moveIdx]

    states[stateIdxs,zIdxs] = states[stateIdxs,swapZIdxs]
    states[stateIdxs,swapZIdxs] = 0

    return(states)

def generate_envs(numStates,scrambleRange,probs=None):
    assert(scrambleRange[0] >= 0)
    scrambs = range(scrambleRange[0],scrambleRange[1]+1)
    legalMoves = ['U','D','L','R']

    states = np.tile(np.expand_dims(solvedState,0),(numStates,1))

    scrambleNums = np.random.choice(scrambs,numStates,p=probs)
    numMoves = np.zeros(numStates)
    while np.max(numMoves < scrambleNums):
        poses = np.where((numMoves < scrambleNums))[0]

        subsetSize = max(len(poses)//len(legalMoves),1)
        poses = np.random.choice(poses,subsetSize)

        move = choice(legalMoves)
        states[poses] = next_state(states[poses], move)
        numMoves[poses] = numMoves[poses] + 1

    return(states,scrambleNums)

print(generate_envs(30, [1,30]))