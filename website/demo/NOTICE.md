# Browser runtime credits

This independent G1 23DoF browser demo uses MuJoCo 3.8.0 (Google DeepMind, Apache-2.0), ONNX Runtime Web 1.23.2 (Microsoft, MIT and included third-party notices), and Three.js 0.180.0 (MIT). The complete license texts are included in `vendor/licenses/` in the runtime archive and in `licenses/browser/` in the source repository.

Robot geometry and physical description originate from Unitree; see the repository's `licenses/` and `docs/provenance.md`. The displayed policies are this project's trained 80k and 104k models, not official SuperMimic weights.

Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. This demo distributes the trained decoder and checkpoint-specific motion encodings for robot control, not the raw BONES-SEED dataset.
