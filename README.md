# Rubiks-CUbe-Reinforcement-Learning

Solving a Rubik's Cube using Deep Reinforcement Learning and Search.

This project is an implementation of the following paper http://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf.

The network was trained using a method named Deep Approximate Value Iteration (DAVI). This estimates the number of moves a scramble is from being solved. A batched weighted A Star search is then done to solve the puzzle. Implementations to solve the 2x2 Cube, 15-Puzzle and 24-Puzzle are also included in the code.
