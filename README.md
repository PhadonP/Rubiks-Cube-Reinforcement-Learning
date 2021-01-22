<p align="center">
  <a href="" rel="noopener">
 <img width=650px src="Readme Images\RubiksCube.gif" alt="CubeDemonstration"></a>
</p>

<h1 align="center">Rubik's Cube Reinforcement Learning</h1>

<p align="center">Solving a Rubik's Cube using Deep Reinforcement Learning and Search.
<br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Results](#results)
- [Using the Code](#usage)
- [Prerequisites](#prerequisites)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

This project is an implementation of the following paper http://deepcube.igb.uci.edu/static/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf.

The network was trained using a method named Deep Approximate Value Iteration (DAVI). This estimates the number of moves a scramble is from being solved. A batched weighted A Star search is then done to solve the puzzle. Implementations to solve the 2x2 Cube, 15-Puzzle and 24-Puzzle are also included in the code.

This project contrasts with many other deep learning projects solving the Rubik's Cube as it is uses pure reinforcement learning, learning without copying other solvers.

For a deep dive into the technical aspects of the project, you can browse through the report in the pdf. Feel free to ask me any questions about using the code.

## 📈 Results <a name = "results"></a>

Each puzzle was tested by testing them on a set of 100 random scrambles. The tests were run on my ultrabook.

<p align="center">
<img src="Readme Images\Results Table.png" width="500" title="Results">
</p>

The results for the 2x2 Cube and 15-Puzzle were extremely good and produced the optimal solution most of the time within a few seconds. The 24-Puzzle averaged 101.88 moves with a search time around 20 seconds, which is roughly 12 moves away from optimal solutions. For the 3x3 Cube, average move count was 41.45 with an average time of 62.51 seconds. This is quite poor compared to the results achieved in the above paper, however computing resources were limited and parallel training was not implemented. 98 out of 100 cubes were solved, which is good compared to many other projects on Github which struggle with scrambles of length 10 or more. Different network architectures and training hyperparameters need to be tried to maximise performance.

Training was accomplished on a Tesla P100 GPU. For the 2x2 Cube and 15-Puzzle, training took around 1 hour. The 24-Puzzle took 1 day. The 3x3 Cube took around 7 days.

<p align="center">
  <a href="" rel="noopener">
 <img width=500px src="Readme Images\15Puzzle.gif" alt="15PuzzleDemonstration"></a>
</p>

## 📖 Usage <a name = "usage"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Training 
Training a new model is done by running the train file.
The config file location has to be written as an argument and it will contain the parameters for training.
Mul

The names of the different networks to be written in the networkType setting are given in the networks/getNetwork.py file. 

    python train.py -c config/cube3.ini -mp True

Training can be restarted by giving a existing network as an argument.

    python train.py -c config/cube3.ini -nt saves\YourNetworkHere.pt -mp True

The run can be tracked while training using tensorboard.

    tensorboard --logdir=runs

### Testing

Testing of a model can be done using the solve file. This file tests the model on a test set with a chosen size and generates performance statistics.

    python solve.py -n saves/YourNetworkHere.pt -c config/cube3.ini 

### Running the GUIs

To run the Sliding Puzzle GUI, run the following line.

    python puzzleNgui.py -n saves/YourNetworkHere.pt -c config/puzzle15.ini

To run the GUI for the Rubik's Cube is a little more complicated. The simulation was written in Javascript and communication was done using Flask as an API.

To run the simulation, you can run a live server on the index.html file.

To connect your network to the simulation, run the following line. Unfortunately, running the network alongside the GUI can slow down the solve time if the API is running on the same computer as the GUI.

    python cubeAPI.py -n2 saves/Your2x2NetworkHere.pt -n3 saves/Your3x3NetworkHere.pt -c2 config/cube2.ini -c3 config/cube3.ini
## ⛏️ Prerequisites <a name = "prerequisites"></a>

This project uses a couple libraries that need to be installed.

1. PyTorch
2. Tensorboard
3. Flask    (Cube API)
4. Three.js    (Cube GUI)
5. Tween.js (Cube GUI)
6. Pygame   (Sliding Puzzle GUI)

### Installing

The python packages can be installed using the following code.

    pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

    python -m pip install -U pygame --user

    pip install -U Flask flask-restful flask-cors

The Javascript modules can be downloaded using npm. These lines should be run while inside the cubeGUI folder.

    npm install --save three
    
    git clone https://github.com/tweenjs/tween.js
    cd tween.js
    npm i .
    npm run build
## 🎉 Acknowledgements <a name = "acknowledgement"></a>

I would like to give thanks to my supervisor for this project, Mehrtash Harandi. Another big thanks has to go to the creators of DeepCubeA, which provided the algorithm of this project.
