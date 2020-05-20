import * as THREE from "./node_modules/three/src/Three.js";
import TWEEN from "./node_modules/tween.js/dist/tween.esm.js";
import { OrbitControls } from "./node_modules/three/examples/jsm/controls/OrbitControls.js";
import Cube from "./cube.js";
import MouseControls from "./controls/mouseControls.js";
import KeyboardControls from "./controls/keyboardControls.js";

let camera, controls, scene, renderer, cube, mouseControls, keyboardControls;
let solveButton, scrambleButton, switchPuzzleButton, solveSteps, solveMessage;
let rewindButton, backButton, forwardButton, fastForwardButton;
let cubeSize = 3;
init();
animate();

function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeeeeee);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    1,
    200
  );

  camera.position.set(2 * cubeSize, 1.15 * cubeSize, 2 * cubeSize);
  camera.setViewOffset(
    window.innerWidth,
    window.innerHeight,
    window.innerWidth * 0.2,
    0,
    window.innerWidth,
    window.innerHeight
  );
  scene.add(camera);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.rotateSpeed = 1.5;

  cube = new Cube(cubeSize, scene);

  mouseControls = new MouseControls(
    scene,
    camera,
    controls,
    cube,
    resetSolveButtons
  );
  keyboardControls = new KeyboardControls(cube, resetSolveButtons);

  solveButton = document.getElementById("solve");
  scrambleButton = document.getElementById("scramble");
  switchPuzzleButton = document.getElementById("switchPuzzle");
  solveMessage = document.getElementById("solveMessage");

  rewindButton = document.getElementById("rewind");
  backButton = document.getElementById("back");
  forwardButton = document.getElementById("forward");
  fastForwardButton = document.getElementById("fastForward");

  solveButton.addEventListener("click", solve);
  scrambleButton.addEventListener("click", scramble);
  switchPuzzleButton.addEventListener("click", switchPuzzle);
  window.addEventListener("resize", onWindowResize, false);

  rewindButton.addEventListener("click", rewind);
  backButton.addEventListener("click", back);
  forwardButton.addEventListener("click", forward);
  fastForwardButton.addEventListener("click", fastForward);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  cube.checkForMove();
  TWEEN.update();
  renderer.render(scene, camera);
}

function solve() {
  if (cube.animating) {
    solveMessage.innerHTML = "Wait until Cube is Still";
    return;
  }

  const xhr = new XMLHttpRequest();
  const url = "http://127.0.0.1:5000/";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-type", "application/json");

  let data = JSON.stringify({ cubeSize: cubeSize, scramble: cube.state() });
  console.log(data);
  xhr.send(data);

  solveMessage.innerHTML = "Solving...";
  cube.sequencing = true;

  solveButton.disabled = true;
  scrambleButton.disabled = true;
  switchPuzzleButton.disabled = true;

  xhr.onreadystatechange = () => {
    let solveString = `Failed to find Solution`;
    if ((xhr.readyState === 4) & !(xhr.status == 0)) {
      let response = JSON.parse(xhr.responseText);
      let isSolved = response.isSolved;
      if (isSolved) {
        let solveTime = response.solveTime;
        let solve = response.solve;
        solveMessageStr = `Solved in ${solveTime} seconds`;
        solveStepsStr = `${solve.join(" ")}`;
        cube.setSolve(solve);
        if (solve.length > 0) {
          forwardButton.disabled = false;
          fastForwardButton.disabled = false;
        }
      }
    }

    cube.sequencing = false;
    solveMessage.innerHTML = solveMessageStr;
    solveStepsStr.innerHTML = solveStepsStr;

    solveButton.disabled = false;
    scrambleButton.disabled = false;
    switchPuzzleButton.disabled = false;
  };
}

function scramble() {
  let scramble = cube.createScramble(50);
  cube.doScramble(scramble);

  let disableButtons = () => {
    solveButton.disabled = false;
    scrambleButton.disabled = false;
    switchPuzzleButton.disabled = false;
  };

  solveButton.disabled = true;
  scrambleButton.disabled = true;
  switchPuzzleButton.disabled = true;

  cube.sequenceEnd = disableButtons;

  resetSolveButtons();
}

function switchPuzzle() {
  for (let cubie of cube.cubies) {
    scene.remove(cubie.mesh);
    cubie.mesh.geometry.dispose();
    cubie.mesh.material.dispose();
    cubie.mesh = undefined;
  }
  if (cube.size == 3) {
    cube = new Cube(2, scene);
    switchPuzzleButton.innerHTML = "3x3";
  } else if (cube.size == 2) {
    cube = new Cube(3, scene);
    switchPuzzleButton.innerHTML = "2x2";
  }
  mouseControls.cube = cube;
  keyboardControls.cube = cube;

  resetSolveButtons();
}

function resetSolveButtons() {
  rewindButton.disabled = true;
  backButton.disabled = true;
  forwardButton.disabled = true;
  fastForwardButton.disabled = true;
  solveMessage.innerHTML = "";
  solveSteps.innerHTML = "";
}

function rewind() {
  cube.doReverseSequence(cube.solveMoves.slice(0, cube.solveIdx));

  let disableButtons = () => {
    rewindButton.disabled = true;
    backButton.disabled = true;
    forwardButton.disabled = false;
    fastForwardButton.disabled = false;
  };

  rewindButton.disabled = true;
  backButton.disabled = true;
  forwardButton.disabled = true;
  fastForwardButton.disabled = true;

  cube.sequencing = true;
  cube.sequenceCount = cube.solveIdx;
  cube.sequenceEnd = disableButtons;

  cube.solveIdx = 0;
}

function back() {
  if (cube.solveIdx == cube.solveMoves.length - 1) {
    forwardButton.disabled = false;
    fastForwardButton.disabled = false;
  }

  cube.reverseMove(cube.solveMoves[cube.solveIdx - 1]);

  if (cube.solveIdx == 1) {
    rewindButton.disabled = true;
    backButton.disabled = true;
  }

  cube.solveIdx -= 1;
}

function forward() {
  if (cube.solveIdx == 0) {
    rewindButton.disabled = false;
    backButton.disabled = false;
  }

  cube.move(cube.solveMoves[cube.solveIdx]);

  if (cube.solveIdx == cube.solveMoves.length) {
    forwardButton.disabled = true;
    fastForwardButton.disabled = true;
  }

  cube.solveIdx += 1;
}

function fastForward() {
  cube.doSequence(cube.solveMoves.slice(cube.solveIdx));

  let disableButtons = () => {
    rewindButton.disabled = false;
    backButton.disabled = false;
    forwardButton.disabled = true;
    fastForwardButton.disabled = true;
  };

  rewindButton.disabled = true;
  backButton.disabled = true;
  forwardButton.disabled = true;
  fastForwardButton.disabled = true;

  cube.sequencing = true;
  cube.sequenceCount = cube.solveMoves.length - cube.solveIdx;
  cube.sequenceEnd = disableButtons;

  cube.solveIdx = cube.solveMoves.length - 1;
}
