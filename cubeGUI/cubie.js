import * as THREE from "./node_modules/three/src/Three.js";
import RoundedBoxGeometry from "./geometries/roundedBoxGeometry.js";
import RoundedRectGeometry from "./geometries/roundedRectGeometry.js";

class Cubie {
  constructor(x, y, z, offset, stickers) {
    this.pos = [x, y, z];
    this.layerPos = [x + offset, y + offset, z + offset];
    this.offset = offset;
    this.boxSize = 1;
    let geometry = new RoundedBoxGeometry(
      this.boxSize,
      this.boxSize,
      this.boxSize,
      0.1,
      5
    );
    let material = new THREE.MeshBasicMaterial({
      color: 0x282828,
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.position.set(x, y, z);
    this.mesh.updateMatrix();
    this.stickers = stickers;
    this.createStickers(stickers);
  }

  setPosition = (x, y, z) => {
    this.pos = [x, y, z];
    this.layerPos = [x + this.offset, y + this.offset, z + this.offset];
    this.mesh.position.set(x, y, z);
  };

  createStickers = (stickers) => {
    for (let sticker of stickers) {
      let [, colour, axis, side] = sticker;
      this.createSticker(colour, axis, side);
    }
  };

  createSticker = (colour, axis, side) => {
    let stickerGeometry = new RoundedRectGeometry(0.85, 0.85, 0.1);
    let stickerMaterial = new THREE.MeshBasicMaterial({
      color: colour,
      side: THREE.DoubleSide,
    });

    let stickerMesh = new THREE.Mesh(stickerGeometry, stickerMaterial);

    if (axis == "x") {
      stickerMesh.translateX((side * (this.boxSize + 0.001)) / 2);
      stickerMesh.rotation.set(0, Math.PI / 2, 0);
    } else if (axis == "y") {
      stickerMesh.translateY((side * (this.boxSize + 0.001)) / 2);
      stickerMesh.rotation.set(Math.PI / 2, 0, 0);
    } else if (axis == "z") {
      stickerMesh.translateZ((side * (this.boxSize + 0.001)) / 2);
    }

    this.mesh.add(stickerMesh);
  };
}

export default Cubie;
