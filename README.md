# highway-env

A collection of environments for *autonomous driving* and tactical decision-making tasks

<p align="center">
    <img src="../gh-media/docs/media/highway-env.gif?raw=true"><br/>
    <em>An episode of one of the environments available in highway-env.</em>
</p>

[![Build Status](https://travis-ci.org/eleurent/highway-env.svg?branch=master)](https://travis-ci.org/eleurent/highway-env)
[![Docs Status](https://img.shields.io/badge/docs-passing-green.svg)](https://eleurent.github.io/highway-env/)

## [:ledger: Try it on Google Colab!](scripts)

## Installation

`pip install --user git+https://github.com/eleurent/highway-env`

## Usage

```python
import highway_env

env = gym.make("highway-v0")

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, _ = env.step(action)
    env.render()
```

## Citing

If you use the project in your work, please consider citing it with:
```
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

## The environments

### Highway

```python
env = gym.make("highway-v0")
```

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles.
The agent's objective is to reach a high velocity while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

<p align="center">
    <img src="../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>


### Merge

```python
env = gym.make("merge-v0")
```

In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high velocity while making room for the vehicles so that they can safely merge in the traffic.

<p align="center">
    <img src="../gh-media/docs/media/merge-env.gif?raw=true"><br/>
    <em>The merge-v0 environment.</em>
</p>

### Roundabout

```python
env = gym.make("roundabout-v0")
```

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

<p align="center">
    <img src="../gh-media/docs/media/roundabout-env.gif?raw=true"><br/>
    <em>The roundabout-v0 environment.</em>
</p>

### Parking

```python
env = gym.make("parking-v0")
```

A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.

<p align="center">
    <img src="../gh-media/docs/media/parking-env.gif?raw=true"><br/>
    <em>The parking-v0 environment.</em>
</p>

### Intersection

```python
env = gym.make("intersection-v0")
```

An intersection negotiation task with dense traffic.

<p align="center">
    <img src="../gh-media/docs/media/intersection-env.gif?raw=true"><br/>
    <em>The intersection-v0 environment.</em>
</p>

## The framework

New highway driving environments can easily be made from a set of building blocks.

### Roads

A `Road` is composed of a `RoadNetwork` and a list of `Vehicles`. The `RoadNetwork` describes the topology of the road infrastructure as a graph, where edges represent lanes and nodes represent intersections. For every edge, the corresponding lane geometry is stored in a `Lane` object as a parametrized center line curve, providing a local coordinate system.

### Vehicle kinematics

The vehicles kinematics are represented in the `Vehicle` class by a _Kinematic Bicycle Model_.

<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;x=v\cos\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;x=v\cos\psi" title="\dot x=v\cos\psi" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;y=v\sin\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;y=v\sin\psi" title="\dot y=v\sin\psi" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot&space;v=a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot&space;v=a" title="\dot v=a" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot\psi=\frac{v}{l}\tan\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot\psi=\frac{v}{l}\tan\beta" title="\dot\psi=\frac{v}{l}\tan\beta" /></a>

Where *(x, y)* is the vehicle position, *v* its forward velocity and *psi* its heading.
*a* is the acceleration command and *beta* is the slip angle at the center of gravity, used as a steering command.

### Control

The `ControlledVehicle` class implements a low-level controller on top of a `Vehicle`, allowing to track a given target velocity and follow a target lane.

### Behaviours

The vehicles populating the highway follow simple and realistic behaviours that dictate how they accelerate and steer on the road.

In the `IDMVehicle` class,
* Longitudinal Model: the acceleration of the vehicle is given by the Intelligent Driver Model (IDM) from [(Treiber et al, 2000)](https://arxiv.org/abs/cond-mat/0002177).
* Lateral Model: the discrete lane change decisions are given by the MOBIL model from [(Kesting et al, 2007)](https://www.researchgate.net/publication/239439179_General_Lane-Changing_Model_MOBIL_for_Car-Following_Models).

In the `LinearVehicle` class, the longitudinal and lateral behaviours are defined as linear weightings of several features, such as the distance and velocity difference to the leading vehicle.

## The agents

Agents solving the `highway-env` environments are available in the [RL-Agents](https://github.com/eleurent/rl-agents) repository.

`pip install --user git+https://github.com/eleurent/rl-agents`

### [Deep Q-Network](https://github.com/eleurent/rl-agents/tree/master/rl_agents/agents/dqn)


<p align="center">
    <img src="../gh-media/docs/media/dqn.gif?raw=true"><br/>
    <em>The DQN agent solving highway-v0.</em>
</p>

This model-free value-based reinforcement learning agent performs Q-learning with function approximation, using a neural network to represent the state-action value function Q.

### [Deep Deterministic Policy Gradient](https://github.com/openai/baselines/tree/master/baselines/her)

<p align="center">
    <img src="../gh-media/docs/media/ddpg.gif?raw=true"><br/>
    <em>The DDPG agent solving parking-v0.</em>
</p>

This model-free policy-based reinforcement learning agent is optimized directly by gradient ascent. It uses Hindsight Experience Replay to efficiently learn how to solve a goal-conditioned task.

### [Value Iteration](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/dynamic_programming/value_iteration.py)

<p align="center">
    <img src="../gh-media/docs/media/ttcvi.gif?raw=true"><br/>
    <em>The Value Iteration agent solving highway-v0.</em>
</p>

The Value Iteration is only compatible with finite discrete MDPs, so the environment is first approximated by a [finite-mdp environment](https://github.com/eleurent/finite-mdp) using `env.to_finite_mdp()`. This simplified state representation describes the nearby traffic in terms of predicted Time-To-Collision (TTC) on each lane of the road. The transition model is simplistic and assumes that each vehicle will keep driving at a constant velocity without changing lanes. This model bias can be a source of mistakes.

The agent then performs a Value Iteration to compute the corresponding optimal state-value function.


### [Monte-Carlo Tree Search](https://github.com/eleurent/rl-agents/blob/master/rl_agents/agents/tree_search/mcts.py)

This agent leverages a transition and reward models to perform a stochastic tree search [(Coulom, 2006)](https://hal.inria.fr/inria-00116992/document) of the optimal trajectory. No particular assumption is required on the state representation or transition model.

<p align="center">
    <img src="../gh-media/docs/media/mcts.gif?raw=true"><br/>
    <em>The MCTS agent solving highway-v0.</em>
</p>
