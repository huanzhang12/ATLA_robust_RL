# Robust Reinforcement Learning with Alternating Training of Learned Adversaries (ATLA)

This repository contains a reference implementation for alternating training of
learned adversaries (ATLA) for robust reinforcement learning against
adversarial attacks on state observations. 

Our ATLA training procedure can be somewhat analogous to "adversarial training"
for supervised learning, but we are based on the [state-adversarial Markov
decision process (SA-MDP)](https://arxiv.org/pdf/2003.08938) which
characterizes the *optimal* adversarial attack for RL agents.  During training,
we learn an adversary along with the agent following the *optimal* attack
formulation. The agent must defeat this strong adversary during training time,
thus becomes robust against a wide range of strong attacks during test time.
Previous approaches were not based on SA-MDP and used gradient based attack
heuristics during training which are not strong enough, and they become
vulnerable under strong test time attacks.

Following SA-MDP, we can find the optimal adversarial attack which achieves the
**lowest possible reward** given an agent and an environment by solving a
transformed MDP. This can be analogous to the **minimal adversarial examples**
(which [can be found via MILP/SMT
solvers](https://arxiv.org/pdf/1709.10207.pdf)) in classification problems.  In
DRL setting, this MDP can be solved using any DRL algorithms such as PPO.  We
demonstrate that adversarial attacks based on the optimal adversary framework
can be significantly stronger than previously proposed stronger attack (see
[examples below](#optimal-adversarial-attack-and-atla-ppo-demo)). The optimal
attack framework can be useful for evaluating the robustness of RL agents for
defense techniques developed in future.

Details on the optimal adversarial attack to RL and the ATLA training framework
can be found in our paper:

"**Robust Reinforcement Learning on State Observations with Learned Optimal Adversary**",  
by [Huan Zhang](http://huan-zhang.com) (UCLA), [Hongge Chen](https://scholar.google.com/citations?user=KFtsQvIAAAAJ&hl=en) (MIT), 
[Duane Boning](https://www-mtl.mit.edu/wpmu/researchgroupsboning/boning/) (MIT), and [Cho-Jui Hsieh](http://web.cs.ucla.edu/~chohsieh/) (UCLA) (\* Equal contribution)  
ICLR 2021. [(Paper PDF)](https://arxiv.org/pdf/2101.08452.pdf)

Our code is based on the SA-PPO robust reinforcement learning codebase:
[huanzhang12/SA_PPO](https://github.com/huanzhang12/SA_PPO).


## Optimal Adversarial Attack and ATLA-PPO Demo

In our paper we first show that we can learn an adversary under the *optimal*
adversarial attack setting in SA-MDP. This allows us to obtain significantly
stronger adversaries to attack RL agents: while previous strong attacks can
make the agents fail to move, our learned adversary can lead the agent into
**moving towards the opposite direction**, obtaining **a large negative
reward**.  Further more, training with this learned strong adversary using our
ATLA framework allows the agent to be robust under strong adversarial attacks.

| | Vanilla PPO <br> No attacks | Vanilla PPO under <br> Robust Sarsa (RS) <br> attack | Vanilla PPO under <br> Learned Optimal attack | Our ATLA-PPO under <br> Learned Optimal attack <br> (strongest attack)|
|:--:|:--:| :--:| :--:| :--:|
| Ant-v2 | ![ant_ppo_natural_5358.gif](/assets/ant_ppo_natural_5358.gif) | ![ant_ppo_rs_attack_63.gif](/assets/ant_ppo_rs_attack_63.gif) | ![ant_ppo_optimal_attack_-1141.gif](/assets/ant_ppo_optimal_attack_-1141.gif) | ![ant_atla_ppo_optimal_attack_3835.gif](/assets/ant_atla_ppo_optimal_attack_3835.gif) |
|Episode <br> rewards | **5358** <br> Moving right ➡️ | **63** <br> Not moving 🛑 | **-1141** <br> Moving left ◀️  <br> (opposite to the goal)| **3835** <br> Moving right ➡️  |
| **HalfCheetah-v2** | ![halfcheetah_ppo_natural_7094.gif](/assets/halfcheetah_ppo_natural_7094.gif) | ![halfcheetah_ppo_rs_attack_85.gif](/assets/halfcheetah_ppo_rs_attack_85.gif) | ![halfcheetah_ppo_optimal_attack_-743.gif](/assets/halfcheetah_ppo_optimal_attack_-743.gif) | ![halfcheetah_ppo_natural_7094.gif](/assets/halfcheetah_atla_ppo_optimal_attack_5250.gif) |
|Episode <br> rewards | **7094** <br> Moving right ➡️ | **85** <br> Not moving 🛑 | **-743** <br> Moving left ◀️  <br> (opposite to the goal) | **5250** <br> Moving right ➡️  |

## Setup

First clone this repository and install necessary Python packages:

```bash
git submodule update --init
pip install -r requirements.txt
sudo apt install parallel  # Only necessary for running the optimal attack experiments.
cd src  # All code files are in the src/ folder
```

Note that you need to install MuJoCo 1.5 first to use the OpenAI Gym environments.
See [here](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#requirements)
for instructions.

## Pretrained agents

We release pretrained agents for all settings evaluated in our paper. These
pretrained agents can be found in `src/models/atla_release`, with six
subdirectories corresponding to six settings evaluated in our paper. Inside
each folder, you can find agent models (starting with `model-`) as well as the
adversary we learned for the optimal adversarial attacks (starting with
`attack-`). We will show how to load these models in later sections. The
performance of our pretrained agents are reported below. Here we report our
strongest ATLA-PPO (LSTAM + SA-Reg) method and a strong baseline SA-PPO, as
well as vanilla PPO without robust training.  We report their natural episode
rewards without attack as well as episode rewards under our proposed optimal
attack. For full results with more baselines please checkout [our paper](https://arxiv.org/pdf/2101.08452.pdf).

| Environment      | Evaluation    | Vanilla PPO | SA-PPO         | ATLA-PPO (LSTM + SA-Reg) |
|------------------|---------------|-------------|----------------|--------------------------|
| Ant-v2           | No attack     |  5687.0    |      4292.1    |       5358.7             |
|                  | Strongest Attack|   -871.7     |    2511.0      |         **3764.5**           |
| HalfCheetah-v2   | No attack     |      7116.7       |       3631.5         |             6156.5             |
|                  | Strongest Attack|      -660.5       |      3027.9          |              **5058.2**             |
| Hopper-v2      | No attack     |        3167.3     |        3704.5        |             3291.2             |
|                  | Strongest Attack|        636.4     |        1076.3        |            **1771.9**              |
| Walker2d-v2        | No attack     |      4471.7       |        4486.6        |               3841.7           |
|                  | Strongest Attack|    1085.5  |         2907.7       |              **3662.9**            |


Note that reinforcement learning algorithms typically have large variance
across training runs. Thus, we repeatedly train each agent configuration 21
times, and rank them with their average cumulative rewards over 50 episodes
under the **strongest (best) attack** (among 6 attacks used). The pretrained
agents are the ones with **median robustness** (median episode reward under the
strongest attack) rather than the best ones. When compared to our work, **it is
important to train each agent repeatedly at least 10 times and report the
median agent, rather than the best**. Additionally, for robust sarsa (RS)
attack and the proposed optimal attack, a large number of attack
parameters are searched and we choose the strongest adversary among them. See
[the section below](#optimal-attack-to-deep-reinforcement-learning).


The pretrained agents can be evaluated using `test.py` (see the next sections
for more usage details). For example,


```bash
# Ant agents.
## Vanilla PPO:
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --deterministic
## SA-PPO:
python test.py --config-path config_ant_sappo_convex.json --load-model models/atla_release/SAPPO/model-sappo-convex-ant.model --deterministic
## Vanilla LSTM:
python test.py --config-path config_ant_vanilla_ppo_lstm.json --load-model models/atla_release/LSTM-PPO/model-lstm-ppo-ant.model --deterministic
## ATLA PPO (MLP):
python test.py --config-path config_ant_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-ant.model --deterministic
## ATLA PPO (LSTM):
python test.py --config-path config_ant_atla_ppo_lstm.json --load-model models/atla_release/ATLA-LSTM-PPO/model-lstm-atla-ppo-ant.model --deterministic
## ATLA PPO (LSTM+SA Reg):
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --deterministic
```

Note that the **--deterministic** switch is important, which disables
stochastic actions for evaluation. You can change `ant` to `walker`, `hopper`
or `halfcheetah` in the config file names and agent model file names to try
other environments.


## Optimal Attack to Deep Reinforcement Learning
### Train a Single Optimal Attack Adversary
To run optimal attack, we set `--mode` to `adv_ppo` and set `--ppo-lr-adam` to
zero.  This essentially runs our ATLA training but with the learning rate of
the agent model set to 0, so this will learn the adversary only. The learning rate of
the adversary policy network can be set via `--adv-ppo-lr-adam`, the learning
rate of the value network can be set via `--adv-val-lr`, the entropy
regularizer of the adversary can be set via `--adv-entropy-coeff`, the clipping
epsilon for PPO optimizer for the adversary can be set via `--adv-clip-eps`.

```bash
# Note: this is for illustration only. We must correctly choose hyperparameters for the adversary, typically via a hyperparameter search.
python run.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --mode adv_ppo --ppo-lr-adam 0.0 --adv-ppo-lr-adam 3e-5 --adv-val-lr 3e-5 --adv-entropy-coeff 0.0 --adv-clip-eps 0.4
```
This will save an experiment folder at `vanilla_ppo_ant/agents/YOUR_EXP_ID`, where `YOUR_EXP_ID` is a randomly generated experiment ID, for example `e908a9f3-0616-4385-a256-4cdea5640725`. You can extract the best model from this folder by running

```bash
python get_best_pickle.py vanilla_ppo_ant/agents/YOUR_EXP_ID
```
which will generate an adversary model `best_model.YOUR_EXP_ID.model`, for example `best_model.e908a9f3.model`.

Then you can evaluate this trained adversary by running

```bash
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network best_model.YOUR_EXP_ID.model
```

### Finding the Best Optimal Attack Adversary
The above command only trains and tests one adversary using one set of adversary
hyperparameters.  Since the learning of this optimal adversary is also an RL
problem (solved using PPO), to obtain the best attack results and evaluate the
true robustness of an agent model, we need to train the adversary using multiple sets
of hyperparameters and take the strongest (best) adversary. We provide scripts
to easily scan the hyperparameters of the adversary and run each set of
hyperparameters in parallel:

```bash
cd ../configs
# This will generate 216 config files inside agent_configs_attack_ppo_ant.
# Modify attack_ppo_ant_scan.py to change the hyperparameters for the grid search.
# Typically, for a different environment you need a different set of hyperparameters for searching.
python attack_ppo_ant_scan.py
cd ../src
# This command will run 216 configurations using all available CPUs. 

# You can also use "-t " to control the number of threads if you don't want to use all CPUs.
python run_agents.py ../configs/agent_configs_attack_ppo_ant_scan/ --out-dir-prefix=../configs/agents_attack_ppo_ant_scan/attack_ppo_ant/agents > attack_ant_scan.log

```

To test all the optimal attack adversaries after the above training command
finishes, simply run the evaluation script:

```bash
bash example_evaluate_optimal_attack.sh
```

Note that you will need to change the line starting with with `scan_exp_folder`
in `example_evaluate_optimal_attack.sh` to run evaluation of the learned
optimal attack adversaries for another environment or results in another
folder. You need to change that line to:

```bash
scan_exp_folder <config file> <path to trained optimal attack adversarial> <path to the victim agent model> $semaphorename
```

This script will run adversary evaluation in parallel (the "GNU parallel" tools
are required), and will generate a log file
`attack_scan/optatk_deterministic.log` containing attack results in each
experiment id folder. After the above command finishes, you can use
`parse_optimal_attack_results.py` to parse the logs and get the best (strongest)
attack result with lowest agent reward:

```
python parse_optimal_attack_results.py ../configs/agents_attack_ppo_ant_scan/attack_ppo_ant/agents
```

If you would like to conduct optimal adversarial attack, it is important to use
a hyperparameter search scheme demonstrated above as the attack itself is a RL
problem and can be sensitive to hyperparameters. To evaluation the true
robustness of an agent, finding the best optimal attack adversary is necessary.

### Pretrained Adversaries for All Agents

We provide optimal attack adversaries for all agents we released.  To
test a pretrained optimal attack adversary we provide, run `test.py` with
the `--attack-advpolicy-network` option:

```bash
# Ant Agents.
## Vanilla PPO:
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/PPO/attack-ppo-ant.model
## SA-PPO:
python test.py --config-path config_ant_sappo_convex.json --load-model models/atla_release/SAPPO/model-sappo-convex-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/SAPPO/attack-sappo-convex-ant.model
## Vanilla LSTM:
python test.py --config-path config_ant_vanilla_ppo_lstm.json --load-model models/atla_release/LSTM-PPO/model-lstm-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/LSTM-PPO/attack-lstm-ppo-ant.model
## ATLA PPO (MLP):
python test.py --config-path config_ant_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/ATLA-PPO/attack-atla-ppo-ant.model
## ATLA PPO (LSTM):
python test.py --config-path config_ant_atla_ppo_lstm.json --load-model models/atla_release/ATLA-LSTM-PPO/model-lstm-atla-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/ATLA-LSTM-PPO/attack-lstm-atla-ppo-ant.model
## ATLA PPO (LSTM+SA Reg):
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/ATLA-LSTM-SAPPO/attack-atla-lstm-sappo-ant.model
```

You can change `ant` to `walker`, `hopper` or `halfcheetah` in the config file
names, agent and adversary model file names to try other environments.


## Agent Training with Learned Optimal Adversaries (our ATLA framework)

To train a agent, use `run.py` in `src` folder and specify a configuration file
path.  Several configuration files are provided in the `src` folder, with
filenames starting with `config`. For example:

Halfcheetah vanilla PPO (MLP) training:

```bash
python run.py --config-path config_halfcheetah_vanilla_ppo.json
```

HalfCheetah vanilla PPO (LSTM) training:

```bash
python run.py --config-path config_halfcheetah_vanilla_ppo_lstm.json
```

HalfCheetah ATLA (MLP) training:

```bash
python run.py --config-path config_halfcheetah_atla_ppo.json
```

HalfCheetah ATLA (LSTM) training:

```bash
python run.py --config-path config_halfcheetah_atla_ppo_lstm.json
```

HalfCheetah ATLA (LSTM) training with state-adversarial regularizer (this is
the best method):

```bash
python run.py --config-path config_halfcheetah_atla_lstm_sappo.json
```

Change `halfcheetah` to `ant`, `hopper` or `walker` to run other environments.

Training results will be saved to a directory specified by the `out_dir`
parameter in the json file. For example, for ATLA (LSTM) training with state-adversarial regularizer
it is `robust_atla_ppo_lstm_halfcheetah`.
To allow multiple runs, each experiment is
assigned a unique experiment ID (e.g., `2fd2da2c-fce2-4667-abd5-274b5579043a`),
which is saved as a folder under `out_dir` (e.g.,
`robust_atla_ppo_lstm_halfcheetah/agents/2fd2da2c-fce2-4667-abd5-274b5579043a`).

Then the agent can be evaluated using `test.py`.  For example:

```bash
# Change the --exp-id to match the folder name in robust_atla_ppo_lstm_halfcheetah/agents/
python test.py --config-path config_halfcheetah_atla_lstm_sappo.json --exp-id YOUR_EXP_ID --deterministic
```

You should expect a cumulative reward (mean over 50 episodes) over 5000 for most methods.


## Agent Evaluation Under Attacks

We implemented random attack, critic based attack and our proposed Robust Sarsa
(RS) and maximal action difference (MAD) attacks.


### Optimal Adversarial Attack

Please see [this section](#optimal-attack-to-deep-reinforcement-learning) for
more details on how to run our proposed optimal adversarial attack. This is
the strongest attack so far and is strongly recommended for evaluating the
robustness of RL defense algorithms.


### Robust Sarsa (RS) Attack

In our Robust Sarsa attack, we first learn a *robust* value function for the
policy under evaluation. Then, we attack the policy using this robust value
function. The first step for RS attack is to train a robust value function
(we use the Ant environment as an example):

```bash
# Step 1:
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --sarsa-enable --sarsa-model-path sarsa_ant_vanilla.model
```

The above training step is usually very fast (e.g., a few minutes).  The value
function will be saved in `sarsa_ant_vanilla.model`. Then it can be used for
attack:

```bash
# Step 2:
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method sarsa --attack-sarsa-network sarsa_ant_vanilla.model --deterministic
```

The L infinity norm for the attack is set by the `--attack-eps` parameter (for
different environments, you will need a different epsilon for attack, see Table
2 in our paper).  The reported mean reward over 50 episodes should less than
500 (reward without attack is over 5000). In contrast, our ATLA-PPO (LSTM +
SA-Reg) robust agent has a reward of over 4000 even under this specific
attack:

```bash
# Train a robust value function.
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --sarsa-enable --sarsa-model-path sarsa_ant_atla_lstm_sappo.model
# Attack using the robust value function.
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --attack-eps=0.15 --attack-method sarsa --attack-sarsa-network sarsa_ant_atla_lstm_sappo.model --deterministic
```

The Robust Sarsa attack has two hyperparameters for robustness regularization
(`--sarsa-eps` and `--sarsa-reg`) to build the robust value function.  Although
the default settings generally work well, for a comprehensive robustness
evaluation it is recommended to run Robust Sarsa attack under different
hyperparameters and choose the best attack (the lowest reward) as the final result.
We provide a script, `scan_attacks.sh` for the purpose of comprehensive
adversarial evaluation:

```bash
# You need to install GNU parall first: sudo apt install parallel
source scan_attacks.sh
# Usage: scan_attacks model_path config_path output_dir_path
scan_attacks models/atla_release/PPO/model-ppo-ant.model config_ant_vanilla_ppo.json sarsa_ant_vanilla_ppo_result
```

In this above example, you should see `minimum RS attack reward (deterministic
action)` reported by the script to be below 300. For your convenience,
the `scan_attacks.sh` script will also run many other attacks including
the MAD attack, critic attack and random attack. Robust sarsa attack
is usually the strongest one among them.

Note: the learning rate of the Sarsa model can be changed by `--val-lr`. The
default value should be good for attacks the provided environments (with
normalized reward). However, if you want to use this attack on a different
environment, this learning rate can be important as the reward maybe
unnormalized (some environment returns large rewards so the Q values are
larger, and a larger `--val-lr` is needed). The rule of thumb is to always
checking the training logs of these Sarsa models - make sure the Q loss has
been reduced sufficiently (close to 0) at the end of training.

### Maximal Action Difference (MAD) Attack

We additionally propose a maximal action difference (MAD) attack where we
attempt to maximize the KL divergence between original action and perturbed
action. It can be invoked by setting `--attack-method` to `action`. For
example:

```bash
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method action --deterministic
```

The reported mean reward over 50 episodes should be around 1500 (this attack is
weaker than the Robust Sarsa attack in this case).  In contrast, our ATLA-PPO (LSTM + SA-Reg) robust agent is more resistant to MAD attack, achieving a reward over 5000.

```bash
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --attack-eps=0.15 --attack-method action --deterministic
```

We additionally provide a combined attack of RS+MAD, which can be invoked by
setting `--attack-method` to `sarsa+action`, and the combination ratio can be
set via `--attack-sarsa-action-ratio`, a number between 0 and 1.

### Critic based attack and random attack

Critic based attack and random attack can be used by setting `--attack-method`
to `critic` and `random`, respectively.  These attacks are relatively weak and
not suitable for evaluating the robustness of PPO agents.

```bash
# Critic based attack (Pattanaik et al.)
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method critic --deterministic
# Random attack (uniform noise)
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method random --deterministic
```

In this case, under critic or random attack the agent reward is still over 5000, which means that these attacks
are not very effective for this specific environment.

### Snooping attack

In this repository, we also implemented an imitation learning-based Snooping
attack propoased by [Inkawhich et al.](https://arxiv.org/pdf/1905.11832.pdf).
In this attack, we first learn a new agent from the policy under evaluation.
Then, we use the gradient information of the new agent to attack the original
policy. The first step for Snooping attack is to train a new imitation agent
by observing ("snooping") how the original agent behaves:

```bash
# Step 1:
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --imit-enable --imit-model-path imit_ant_vanilla.model
```

The above training step is usually very fast (e.g., a few minutes).  The new
agent model will be saved in `imit_ant_vanilla.model`. Then it is loaded to
conduct the Snooping attack:

```bash
# Step 2:
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method action+imit --imit-model-path imit_ant_vanilla.model --deterministic
```

Note that snooping attack is a blackbox attack (does not require the gradient
of the agent policy or interaction with the agent), so it is usually weaker
than other whitebox attacks. In the above example, it should achieve an
average episode reward of roughly 3000.
