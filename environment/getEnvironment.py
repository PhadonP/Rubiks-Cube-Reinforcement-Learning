def getEnvironment(environment):
    if environment == "puzzleN":
        from environment.PuzzleN import PuzzleN
        return PuzzleN
    elif environment == "cubeN":
        from environment.CubeN import CubeN
        return CubeN
    else:
        ValueError("Invalid Puzzle")
