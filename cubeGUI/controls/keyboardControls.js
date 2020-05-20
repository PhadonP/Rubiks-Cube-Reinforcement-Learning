class keyboardControls {
  constructor(cube, resetSolveButtons) {
    this.cube = cube;
    this.resetSolveButtons = resetSolveButtons;
    document.addEventListener("keydown", this.onKeyPress);
  }

  onKeyPress = (event) => {
    if (this.cube.sequencing) {
      return;
    } else if (this.cube.solving) {
      this.cube.solving = false;
      this.resetSolveButtons();
    }

    switch (event.key) {
      case "r":
        this.cube.move("R");
        break;
      case "R":
        this.cube.move("R'");
        break;
      case "u":
        this.cube.move("U");
        break;
      case "U":
        this.cube.move("U'");
        break;
      case "f":
        this.cube.move("F");
        break;
      case "F":
        this.cube.move("F'");
        break;
      case "l":
        this.cube.move("L");
        break;
      case "L":
        this.cube.move("L'");
        break;
      case "d":
        this.cube.move("D");
        break;
      case "D":
        this.cube.move("D'");
        break;
      case "b":
        this.cube.move("B");
        break;
      case "B":
        this.cube.move("B'");
        break;
    }
  };
}

export default keyboardControls;
