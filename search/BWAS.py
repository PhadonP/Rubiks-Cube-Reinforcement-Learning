import time
import torch
from queue import PriorityQueue
from search.node import Node
from networks.PuzzleNet import PuzzleNet
from networks.CubeNet import CubeNet


def batchedWeightedAStarSearch(
    scramble,
    depthWeight,
    numParallel,
    env,
    heuristicFn,
    device,
    maxSearchItr,
    queue=None,
):

    openNodes = PriorityQueue()
    closedNodes = dict()

    root = Node(scramble, None, None, 0, 0, env.checkIfSolvedSingle(scramble))
    openNodes.put((root.cost, id(root), root))
    closedNodes[hash(root)] = root

    searchItr = 1
    numNodesGenerated = 1
    isSolved = False

    startTime = time.time()

    with torch.no_grad():
        while not isSolved and searchItr <= maxSearchItr:

            numGet = min(openNodes.qsize(), numParallel)
            currNodes = []

            for _ in range(numGet):
                node = openNodes.get()[2]
                currNodes.append(node)
                if node.isSolved:
                    isSolved = True
                    solvedNode = node

            currStates = [node.state for node in currNodes]
            currStates = torch.stack(currStates)

            childrenStates, valChildrenStates, goalChildren = env.exploreNextStates(
                currStates
            )

            children = []
            depths = []

            for i in range(len(currNodes)):
                parent = currNodes[i]
                for j in range(len(childrenStates[i])):
                    if valChildrenStates[i][j]:
                        action = env.NextStateSpotToAction(j)
                        depths.append(parent.depth + 1)
                        children.append(
                            Node(
                                childrenStates[i][j],
                                parent,
                                action,
                                parent.depth + 1,
                                0,
                                goalChildren[i][j],
                            )
                        )

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

            childrenStates = childrenStates[valChildrenStates][nodesToAddIdx].to(device)

            if isinstance(heuristicFn, PuzzleNet) or isinstance(heuristicFn, CubeNet):
                childrenStates = env.oneHotEncoding(childrenStates)

            hValue = heuristicFn(childrenStates)

            bestHValue = min(hValue)
            costs = hValue + depthWeight * depths

            for i in range(len(children)):
                children[i].cost = costs[i]
                openNodes.put((children[i].cost, id(children[i]), children[i]))

            numNodesGenerated += len(children)

            print("Search Itr: %i" % searchItr)
            print("Best Heuristic Function Value: %.2f" % bestHValue)

            searchItr += 1

    searchTime = time.time() - startTime

    if isSolved:

        moves = []

        node = solvedNode

        while node.depth > 0:
            moves.append(node.parentMove)
            node = node.parent

        moves = moves[::-1]

    else:
        moves = None
    if queue:
        queue.put((moves, numNodesGenerated, searchItr, isSolved, searchTime))
    else:
        return moves, numNodesGenerated, searchItr, isSolved, searchTime
