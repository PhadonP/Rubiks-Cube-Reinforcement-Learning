def getNetwork(puzzle, networkType):
    if puzzle == "puzzleN":
        if networkType == "conv":
            from networks.PuzzleNetConv import PuzzleNet
            return PuzzleNet
        elif networkType == "simple":
            from networks.PuzzleNetSimple import PuzzleNet
            return PuzzleNet
        elif networkType == "residual":
            from networks.PuzzleNetRes import PuzzleNet
            return PuzzleNet
        elif networkType == "residualSELU":
            from networks.PuzzleNetResSELU import PuzzleNet
            return PuzzleNet
        else:
            ValueError("Invalid Network Type")
    elif puzzle == "cubeN":
        if networkType == "simple":
            from networks.CubeNetSimple import CubeNet
            return CubeNet
        elif networkType == "residual":
            from networks.CubeNetRes import CubeNet
            return CubeNet
        elif networkType == "paper":
            from networks.CubeNetPaper import CubeNet
            return CubeNet
        else:
            ValueError("Invalid Network Type")
    else:
        ValueError("Invalid Puzzle")
