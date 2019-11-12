from __future__ import division, print_function, absolute_import
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle

# for loading target agent
from rl_agents.agents.common.factory import load_agent


class AttackHighWay(AbstractEnv):
    """
        A highway driving environment.

        The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""

    def __init__(self, config=None):
        super(AttackHighWay, self).__init__(config)
        self.target_model = load_agent(config["target_agent_config"], self)
        self.target_model.load(config["target_agent_load"])

    def default_config(self):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,  # [s]
            "initial_spacing": 2,
            "collision_reward": self.COLLISION_REWARD,
            "target_agent_config": "",
            "target_agent_load": ""
        })
        return config

    def reset(self):
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        return super(AttackHighWay, self).reset()

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self._simulate(action)

        obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return obs, reward, terminal, info

    def _simulate(self, action=None):
        """
            Perform several steps of simulation with constant action
        """
        for k in range(int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])):
            if action is not None and \
                    self.time % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"]) == 0:
                # Forward action to the vehicle
                self.vehicle.act(self.ACTIONS[action])

            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def _create_road(self):
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        # target AV
        self.target_vehicle = MDPVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        self.target_vehicle.color = (200, 0, 150)  # purple
        self.road.vehicles.append(self.target_vehicle)

        # attacker: training agent
        self.vehicle = MDPVehicle.create_random(self.road, 25, spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        return 0

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.target_vehicle.crashed or self.steps >= self.config["duration"]

    def _cost(self, action):
        """
            The cost signal is the occurrence of collision
        """
        return float(self.target_vehicle.crashed)


register(
    id='attack-highway-v0',
    entry_point='highway_env.envs:AttackHighWay',
)
