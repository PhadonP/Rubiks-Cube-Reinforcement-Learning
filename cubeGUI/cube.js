import Cubie from "./cubie.js";
import * as THREE from "./node_modules/three/src/Three.js";
import TWEEN from "./node_modules/tween.js/dist/tween.esm.js";
import { CubeCamera } from "./node_modules/three/src/Three.js";

class Cube {
  constructor(size, scene) {
    this.size = size;
    this.scene = scene;
    this.pivot = new THREE.Object3D();
    this.scene.add(this.pivot);

    this.activeLayer = [];
    this.moveQueue = [];
    this.animating = false;
    this.solving = false;
    this.sequencing = false;
    this.sequenceCount = 0;
    this.rotateTime = 250;
    this.scrambleRotateTime = 50;

    this.solveMoves = [];
    this.solveIdx = 0;

    this.colours = {
      U: new THREE.Color("ghostwhite"),
      D: new THREE.Color("yellow"),
      L: new THREE.Color("green"),
      R: new THREE.Color("blue"),
      F: new THREE.Color("red"),
      B: new THREE.Color("darkorange"),
    };

    this.moves = {
      R: [0, this.size - 1, -1],
      "R'": [0, this.size - 1, 1],
      L: [0, 0, 1],
      "L'": [0, 0, -1],
      U: [1, this.size - 1, -1],
      "U'": [1, this.size - 1, 1],
      D: [1, 0, 1],
      "D'": [1, 0, -1],
      F: [2, this.size - 1, -1],
      "F'": [2, this.size - 1, 1],
      B: [2, 0, 1],
      "B'": [2, 0, -1],
    };

    this.offset = (this.size - 1) / 2;
    this.cubies = this.createCubies();

    this.scene.attach(this.pivot);

    for (let cubie of this.cubies) {
      this.scene.attach(cubie.mesh);
    }
  }

  createCubies = () => {
    let cubies = [];
    let stickers = [];
    for (let x = 0; x < this.size; x++) {
      for (let y = 0; y < this.size; y++) {
        for (let z = 0; z < this.size; z++) {
          if (
            (x != 0) &
            (x != this.size - 1) &
            (y != 0) &
            (y != this.size - 1) &
            (z != 0) &
            (z != this.size - 1)
          ) {
            continue;
          }
          stickers = [];
          if (x == 0) {
            stickers.push([2, this.colours.L, "x", -1]);
          }
          if (x == this.size - 1) {
            stickers.push([5, this.colours.R, "x", 1]);
          }
          if (y == 0) {
            stickers.push([3, this.colours.D, "y", -1]);
          }
          if (y == this.size - 1) {
            stickers.push([0, this.colours.U, "y", 1]);
          }
          if (z == 0) {
            stickers.push([4, this.colours.B, "z", -1]);
          }
          if (z == this.size - 1) {
            stickers.push([1, this.colours.F, "z", 1]);
          }
          cubies.push(
            new Cubie(
              x - this.offset,
              y - this.offset,
              z - this.offset,
              this.offset,
              stickers
            )
          );
        }
      }
    }
    return cubies;
  };

  turn = (side, layer, dir) => {
    this.pivot.rotation.set(0, 0, 0);
    this.pivot.updateMatrixWorld();
    for (let cubie of this.cubies) {
      if (cubie.pos[side] == layer - this.offset) {
        this.pivot.attach(cubie.mesh);
        this.activeLayer.push(cubie);
      }
    }

    let x = 0,
      y = 0,
      z = 0;
    if (side == 0) {
      x = 1;
    } else if (side == 1) {
      y = 1;
    } else if (side == 2) {
      z = 1;
    }
    let time;
    if (this.sequencing & !this.solving) {
      time = this.scrambleRotateTime;
    } else {
      time = this.rotateTime;
    }
    let rotation = { x: 0, y: 0, z: 0 };
    this.tween = new TWEEN.Tween(rotation)
      .to(
        {
          x: (x * (dir * Math.PI)) / 2,
          y: (y * (dir * Math.PI)) / 2,
          z: (z * (dir * Math.PI)) / 2,
        },
        time
      )
      .onUpdate(() => {
        this.pivot.rotation.set(rotation.x, rotation.y, rotation.z);
      })
      .onComplete(this.moveEnd);

    this.tween.start();
  };

  moveEnd = () => {
    this.pivot.updateMatrixWorld();
    this.activeLayer.forEach((cubie) => {
      this.scene.attach(cubie.mesh);
      cubie.mesh.rotation.set(
        this.roundAngle(cubie.mesh.rotation.x),
        this.roundAngle(cubie.mesh.rotation.y),
        this.roundAngle(cubie.mesh.rotation.z)
      );
      let roundedPos = this.roundPosition(cubie.mesh.position);
      cubie.setPosition(roundedPos.x, roundedPos.y, roundedPos.z);
    });

    this.activeLayer = [];
    this.animating = false;
    if (this.sequencing) {
      this.sequenceCount--;
    }

    if ((this.sequenceCount == 0) & this.sequencing) {
      this.sequencing = false;
      this.sequenceEnd();
    }

    if (this.solving) {
      solveSteps.innerHTML = 
    }
  };

  checkForMove = () => {
    if (!this.animating & (this.moveQueue.length > 0)) {
      let move = this.moveQueue.shift();
      this.animating = true;
      this.turn(...move);
    }
  };

  roundPosition = (position) => {
    return position.multiplyScalar(2).round().divideScalar(2);
  };

  roundAngle = (angle) => {
    const round = Math.PI / 2;
    return Math.sign(angle) * Math.round(Math.abs(angle) / round) * round;
  };

  doSequence = (sequence) => {
    for (let action of sequence) {
      this.move(action);
    }
  };

  doReverseSequence = (sequence) => {
    for (let action of sequence.reverse()) {
      this.reverseMove(action);
    }
  };

  move = (move) => {
    if (Array.isArray(move)) {
      this.moveQueue.push(move);
    } else {
      this.moveQueue.push(this.moves[move]);
    }
  };

  reverseMove = (move) => {
    let moveNames = Object.keys(this.moves);
    let moveIdx = moveNames.indexOf(move);

    if (moveIdx % 2) {
      moveIdx--;
    } else {
      moveIdx++;
    }

    this.moveQueue.push(this.moves[moveNames[moveIdx]]);
  };

  state = () => {
    let state = Array(this.size * this.size * 6).fill(0);
    for (let cubie of this.cubies) {
      for (let sticker of cubie.stickers) {
        let [number, color, axis, side] = sticker;
        let stickerNormal = new THREE.Vector3();

        if (axis == "x") {
          stickerNormal.set(side, 0, 0);
        }
        if (axis == "y") {
          stickerNormal.set(0, side, 0);
        }
        if (axis == "z") {
          stickerNormal.set(0, 0, side);
        }

        let cubieDirection = new THREE.Quaternion();
        cubie.mesh.getWorldQuaternion(cubieDirection);
        stickerNormal.applyQuaternion(cubieDirection).round();
        stickerNormal.setLength(1);

        let currSide;

        if (stickerNormal.x == 1) {
          currSide = "x";
        } else if (stickerNormal.x == -1) {
          currSide = "-x";
        } else if (stickerNormal.y == 1) {
          currSide = "y";
        } else if (stickerNormal.y == -1) {
          currSide = "-y";
        } else if (stickerNormal.z == 1) {
          currSide = "z";
        } else if (stickerNormal.z == -1) {
          currSide = "-z";
        }

        state[this.stickerIndex(currSide, ...cubie.layerPos)] = number;
      }
    }
    return state;
  };

  stickerIndex = (side, x, y, z) => {
    let size = this.size;
    let numSideStickers = size * size;
    switch (side) {
      case "x":
        return 5 * numSideStickers + (size - y - 1) * size + (size - z - 1);
      case "-x":
        return 2 * numSideStickers + (size - y - 1) * size + z;
      case "y":
        return x + z * size;
      case "-y":
        return 3 * numSideStickers + x + (size - z - 1) * size;
      case "z":
        return numSideStickers + x + (size - y - 1) * size;
      case "-z":
        return 4 * numSideStickers + (size - x - 1) + (size - y - 1) * size;
    }
  };

  createScramble = (length) => {
    let scramble = [];
    let disabledIndex = -5;
    let randIndex;
    for (let i = 0; i < length; i++) {
      do {
        randIndex = Math.floor(Math.random() * Object.keys(this.moves).length);
      } while (randIndex == disabledIndex);

      if (randIndex % 2) {
        disabledIndex = randIndex - 1;
      } else {
        disabledIndex = randIndex + 1;
      }

      let randKey = Object.keys(this.moves)[randIndex];

      scramble.push(this.moves[randKey]);
    }
    return scramble;
  };

  doScramble = (scramble) => {
    this.sequencing = true;
    this.solving = false;
    this.moveQueue.push(...scramble);
    this.sequenceCount = scramble.length;
  };

  setSolve = (solve) => {
    this.solving = true;
    this.solveMoves = solve;
    this.solveIdx = 0;
  };
}

export default Cube;
