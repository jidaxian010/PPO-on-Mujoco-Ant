# PPO-on-Mujoco-Ant

**A. Train and Evaluate**

1. Setup the environment using:
   
   ```
   conda create -n mujoco_rl python=3.10
   conda activate mujoco_rl
   
   # PyTorch (in my case CUDA12.1)
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install "gymnasium[mujoco]"
   pip install matplotlib numpy IPython pygments
   
   # System level
   sudo apt-get update
   sudo apt-get install -y \
      libgl1-mesa-glx \
      libgl1-mesa-dev \
      libosmesa6-dev \
      patchelf
   sudo apt-get install -y xvfb
   ```

2. Train and save model in `/model`:
   
   ```
   python ant_walk.py
   ```

3. Evaluate model and save videos:
   
   ```
   python ant_evaluate.py
   ```

**B. Slides**

[PPO on Mujoco Ant - Google Slides](https://docs.google.com/presentation/d/18fKE9j3BegDGVhvJY5Bzqkh6GKHr4XkDYMg8quF7y4M/edit?usp=sharing)


