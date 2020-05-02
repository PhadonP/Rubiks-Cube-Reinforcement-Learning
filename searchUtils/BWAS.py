import time
import torch
from queue import PriorityQueue
from searchUtils.node import Node


def batchedWeightedAStarSearch(scramble, depthPenalty, numParallel, env, net, device):

    openNodes = PriorityQueue()
    closedNodes = dict()

    root = Node(scramble, None, None, 0, 0, False)
    openNodes.put((root.cost, root))
    closedNodes[hash(root)] = root

    searchItr = 1
    numNodesGenerated = 1
    isSolved = False

    while not isSolved:
        
        numGet = min(openNodes.qsize(), numParallel)
        currNodes = []
        
        for _ in range(numGet):
            node = openNodes.get()[1]
            currNodes.append(node)
            if node.isSolved:
                isSolved = True
                solvedNode = node
        
        currStates = [node.state for node in currNodes]
        currStates = torch.stack(currStates)
        childrenStates, valChildrenStates, goalChildren = env.exploreNextStates(currStates)

        children = []
        depths = []

        for i in range(len(currNodes)):
            parent = currNodes[i]
            for j in range(len(childrenStates[i])):
                if valChildrenStates[i][j]:
                    action = env.NextStateSpotToAction(j)
                    depths.append(parent.depth + 1)
                    children.append(Node(childrenStates[i][j], parent, 
                        action, parent.depth + 1, 0, goalChildren[i][j]))

        nodesToAddIdx = []
        
        for i, child in enumerate(children):
            if hash(child) in closedNodes:
                if closedNodes[hash(child)].depth > child.depth:
                    found = closedNodes.pop(hash(child))
                    found.depth = child.depth
                    found.parent = child.parent
                    found.parentMove = child.parentMove
                    children[i] = found
                    nodesToAddIdx.append(i)
            else:
                closedNodes[hash(node)] = node
                nodesToAddIdx.append(i)
        
        children = [children[i] for i in nodesToAddIdx]
        depths = torch.tensor([depths[i] for i in nodesToAddIdx]).to(device)
        
        childrenOneHot = env.oneHotEncoding(childrenStates[valChildrenStates][nodesToAddIdx]).to(device)

        movesToGo = net(childrenOneHot).squeeze()
        bestMovesToGo = min(movesToGo)
        costs = movesToGo + depthPenalty * depths
        
        for i in range(len(children)):
            children[i].cost = costs[i]
            openNodes.put((children[i].cost, children[i]))

        numNodesGenerated += len(children)
        #####Put Print Statements
        print("Search Itr: %i" % searchItr)
        print("Best Moves to Go: %.2f" % bestMovesToGo)
        
        del(depths)
        del(childrenOneHot)
        torch.cuda.empty_cache()
        searchItr += 1

    moves = ""

    node = solvedNode

    while node.depth > 0:
        moves += node.parentMove
        node = node.parent
    
    moves = moves[::-1]
    
    return moves, numNodesGenerated, searchItr













			


        


