# Acrobot-v1 Reinforcement Learning Agents

This project implements three reinforcement learning agents to solve the [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) environment using:

- Q-Learning (Tabular)
- SARSA (Tabular)
- Deep Q-Networks (DQN with PyTorch)

## Objective

The goal is to train an agent to apply torques to a two-link robot arm to swing it above a certain height. The environment is solved when the arm’s tip reaches the target height.

## Algorithms Included

- **Q-Learning:** Value-based off-policy learning using discretization.
- **SARSA:** Value-based on-policy learning using discretization.
- **DQN:** Deep learning-based Q-function approximation with experience replay and target networks.

## Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `matplotlib`
- `tqdm`
- `pygame`
- **PyTorch with CUDA 11.8**  
  Install manually:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
## Installation

To get started with this project, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/YasserCoconut/Acrobot-v1.git
cd Acrobot-v1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA 11.8 support

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Run the project

Simply execute the main script and follow the prompts to choose your desired algorithm and configuration:

```bash
python main.py
```

You’ll be able to:

- Train or test a model using **Q-Learning**, **SARSA**, or **DQN**
- Choose to load existing models or train from scratch
- Visualize training and testing progress

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to:

- Open issues for bugs or feature requests
- Submit pull requests with enhancements or fixes
- Improve the documentation or examples

### To contribute:

1. **Fork** the repository
2. **Clone** your fork locally  
   ```bash
   git clone https://github.com/YasserCoconut/Acrobot-v1.git
   ```
3. Create a new branch for your changes  
   ```bash
   git checkout -b my-new-feature
   ```
4. Make your changes and **commit** them  
   ```bash
   git commit -m "Add my cool feature"
   ```
5. **Push** your branch  
   ```bash
   git push origin my-new-feature
   ```
6. Open a **Pull Request** on GitHub and describe what you added

## License

This project is licensed under the [MIT License](LICENSE).

> Q-tables (.npy) are not included in this repo due to file size limits. Contact me for the files or train from scratch.
"# Acrobot-v1" 
