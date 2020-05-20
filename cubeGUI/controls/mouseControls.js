import * as THREE from "../node_modules/three/src/Three.js";

class mouseControls {
  constructor(scene, camera, controls, cube, resetSolveButtons) {
    this.scene = scene;
    this.camera = camera;
    this.mouse = new THREE.Vector2();
    this.clickVector = new THREE.Vector3();
    this.raycaster = new THREE.Raycaster();
    this.controls = controls;
    this.cube = cube;
    this.resetSolveButtons = resetSolveButtons;

    window.addEventListener("mousedown", this.onMouseDown);
    window.addEventListener("mouseup", this.onMouseUp);
  }

  onMouseDown = (event) => {
    if (this.cube.animating == true) {
      return;
    }

    this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    let intersects = this.raycaster.intersectObjects(this.scene.children);
    if (intersects.length > 0) {
      this.controls.enableRotate = false;
      this.cubeClicked = intersects[0];
      this.clickVector = this.mouseVector();

      this.faceNormal = this.cubeClicked.face.normal.clone();

      let cubeDirection = new THREE.Quaternion();
      this.cubeClicked.object.getWorldQuaternion(cubeDirection);
      this.faceNormal = this.faceNormal.applyQuaternion(cubeDirection).round();

      if (this.cube.solving) {
        this.cube.solving = false;
        this.resetSolveButtons();
      }
    } else {
      this.cubeClicked = null;
      this.clickVector = null;
    }
  };

  onMouseUp = (event) => {
    this.controls.enableRotate = true;
    if (this.clickVector == null) {
      return;
    }

    this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    let dragVector = this.mouseVector();
    dragVector = dragVector.sub(this.clickVector);

    if (this.maxComponentSize(dragVector) > 0.1) {
      let moveAxis = new THREE.Vector3();
      moveAxis.crossVectors(dragVector, this.faceNormal);
      moveAxis = this.straightenedVec(moveAxis);

      let layer = this.findLayer(moveAxis);
      let axis, dir;
      if (Math.abs(moveAxis.x) > 0) {
        dir = -Math.sign(moveAxis.x);
        axis = 0;
      } else if (Math.abs(moveAxis.y) > 0) {
        dir = -Math.sign(moveAxis.y);
        axis = 1;
      } else {
        dir = -Math.sign(moveAxis.z);
        axis = 2;
      }
      this.cube.move([axis, layer, dir]);
    }
  };

  mouseVector = () => {
    let vector = new THREE.Vector3();
    vector.set(this.mouse.x, this.mouse.y, 1);
    vector.unproject(this.camera);
    vector.normalize();
    return vector;
  };

  maxComponentSize = (vector) => {
    return Math.max(Math.abs(vector.x), Math.abs(vector.y), Math.abs(vector.z));
  };

  straightenedVec = (vector) => {
    let maxComponent = this.maxComponentSize(vector);
    let straightened = new THREE.Vector3();
    straightened.set(0, 0, 0);
    if (Math.abs(vector.x) == maxComponent) {
      straightened.setX(Math.sign(vector.x));
    } else if (Math.abs(vector.y) == maxComponent) {
      straightened.setY(Math.sign(vector.y));
    } else {
      straightened.setZ(Math.sign(vector.z));
    }
    return straightened;
  };

  findLayer = (axis) => {
    if (Math.abs(axis.x) > 0) {
      return this.cubeClicked.object.position.x + this.cube.offset;
    } else if (Math.abs(axis.y) > 0) {
      return this.cubeClicked.object.position.y + this.cube.offset;
    } else {
      return this.cubeClicked.object.position.z + this.cube.offset;
    }
  };
}

export default mouseControls;
