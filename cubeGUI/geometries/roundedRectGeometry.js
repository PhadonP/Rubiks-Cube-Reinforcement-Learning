import * as THREE from "../node_modules/three/src/Three.js";

class roundedRectGeometry extends THREE.ShapeBufferGeometry {
  constructor(width, height, radius) {
    let shape = new THREE.Shape();
    let x = -width / 2;
    let y = -height / 2;

    shape.moveTo(x, y + radius);
    shape.lineTo(x, y + height - radius);
    shape.quadraticCurveTo(x, y + height, x + radius, y + height);
    shape.lineTo(x + width - radius, y + height);
    shape.quadraticCurveTo(
      x + width,
      y + height,
      x + width,
      y + height - radius
    );
    shape.lineTo(x + width, y + radius);
    shape.quadraticCurveTo(x + width, y, x + width - radius, y);
    shape.lineTo(x + radius, y);
    shape.quadraticCurveTo(x, y, x, y + radius);
    super(shape);
  }
}

export default roundedRectGeometry;
