# ant-learning-walk-model

## Learning Walk Model in Ant Navigation

This repository contains the source code and experiment scripts used in our study on modeling the multimodal mechanisms underlying Learning Walks in ants.

## Contents

- `Learning_Walk_Runing.ipynb`: Jupyter notebook used for running simulations and visualizing results.
- `helper_function.py`: Main class implementation for the `LearningWalkAgent` (**original contribution**).
- `World.mat`: Environment setup file used for simulation.
- Modules adapted or reused from Xuelong Sun's [InsectNavigationToolkitModelling](https://github.com/XuelongSun/InsectNavigationToolkitModelling):
  - `image_processing.py`
  - `visual_homing.py`
  - `insect_brain_model.py`
  - `zernike_moment.py`
  - `insect_navigation.py`

## Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/ant-learning-walk-model.git
cd ant-learning-walk-model
jupyter notebook Learning_Walk_Runing.ipynb
```

## License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.
