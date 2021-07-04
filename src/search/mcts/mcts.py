import numpy as np
from copy import copy
import time

from src.processing.load_dataframe import get_legal_actions
from src.search.mcts.selection import UCB

class MCTSNode():

    def __init__(self, salaries, names, available_pos, scores, level:int, lineup:list,
                 empty_positions:list, budget_left:float, nodes:dict):
        self.lineup = lineup
        self.lineup.sort()
        self.budget_left = budget_left
        self.names = names
        self.salaries = salaries
        self.scores = scores
        self.empty_positions = empty_positions
        self.available_pos = available_pos

        self.nodes = nodes
        self.level = level
        self.legal_actions = get_legal_actions(salaries, names, available_pos, 
            empty_positions, self.players_lineup, self.budget_left, self.level)

        self.children = [None for _ in self.legal_actions]
        self.children_visits = np.array([0 for _ in self.legal_actions])
        self.children_values = np.array([0 for _ in self.legal_actions])

    def build_children(self, action:int):
        new_empty_positions = copy(self.empty_positions)
        new_empty_positions[self.level] = 0

        new_lineup = self.lineup + [(self.level, action)]
        new_lineup.sort()

        if self.level + 1 in self.nodes:
            # for node in self.nodes[self.level + 1]:
            #     if node.lineup == new_lineup:
            #         return node
            pass
        else:
            self.nodes[self.level + 1] = []

        cost = self.salaries[action]
        new_budget_left = self.budget_left - cost

        new_node = MCTSNode(self.salaries, self.names, self.available_pos, self.scores,
            self.level + 1, new_lineup, new_empty_positions, new_budget_left, nodes=self.nodes)
        self.nodes[self.level + 1].append(new_node)
        return new_node


    def get_child_by_id(self, child_id):
        child = self.children[child_id]
        if child is None:
            action = self.legal_actions[child_id]
            child = self.build_children(action)
            self.children[child_id] = child
        return child

    @property
    def players_lineup(self):
        return [self.names[player] for _, player in self.lineup]

    @property
    def is_leaf(self) -> bool:
        return self.is_terminal or np.any(self.children_visits <= 0)

    @property
    def is_terminal(self) -> bool:
        return len(self.legal_actions) == 0

    @property
    def value(self) -> bool:
        if not self.is_terminal:
            raise ValueError("Non-terminal nodes should never be evaluated")
        return np.sum([self.scores[i] for _, i in self.lineup])

    def __repr__(self):
        return f"Node({self.budget_left}$, {self.players_lineup})"

    def __eq__(self, other):
        if isinstance(other, list):
            return self.lineup == other
        return self.lineup == other.lineup

def encode_positions(available_positions):
    positions = []
    for available_pos in available_positions:
        poses = available_pos.split('/')
        for pos in poses:
            if pos not in positions:
                positions.append(pos)

    tensor = np.zeros((len(available_positions), len(positions)), dtype=np.uint8)
    for i, available_pos in enumerate(available_positions):
        poses = available_pos.split('/')
        for pos in poses:
            index = positions.index(pos)
            tensor[i, index] = 1
    return positions, tensor

class MCTSTree():

    def __init__(self, dataframe, positions, budget, exploration=1.):
        self.nodes = {}
        self.dataframe = dataframe
        self.player_names = dataframe['PLAYER_NAME'].to_numpy()
        self.salaries = dataframe['SALARY'].to_numpy()
        self.scores = dataframe['FPTS'].to_numpy()
        self.positions, self.available_pos = encode_positions(dataframe['POSITIONS'].to_numpy())
        self.positions_names = positions

        self.empty_positions = np.zeros(len(self.positions), dtype=np.uint8)
        for pos in positions:
            index = self.positions.index(pos)
            self.empty_positions[index] = 1

        self.root = MCTSNode(self.salaries, self.player_names, self.available_pos, self.scores,
            0, [], self.empty_positions, budget, nodes=self.nodes)
        self.nodes[0] = [self.root]
        self.exploration = exploration
        self.best_node = None

    def run(self, n_simulations=1, verbose=1):
        """ Run the MCTS search for a number of simulations """
        running_average_eval = None
        start_time = time.time()
        for sim in range(n_simulations):
            start_node, path, actions_ids = self.select()
            sim_path, sim_actions_ids = self.simulation(start_node)
            reward = self.evaluate(sim_path[-1], verbose)
            self.backpropagate(reward, path[:-1] + sim_path, actions_ids + sim_actions_ids)

            if running_average_eval is None:
                running_average_eval = reward
            else:
                running_average_eval += (100/n_simulations) * (reward - running_average_eval)

            if verbose > 0 and (sim == 0 or (sim+1)%(n_simulations//100)==0):
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(elapsed_time, 60)
                eta = (n_simulations - sim) * elapsed_time / (sim + 1)
                minutes_left, seconds_left = divmod(eta, 60)
                time_print = f"[{minutes:02.0f}:{seconds:02.0f}-" \
                             f"{minutes_left:02.0f}:{seconds_left:02.0f}]"
                n_nodes = {level:len(self.nodes[level]) for level in self.nodes}
                print(f"{sim+1}/{n_simulations} - {time_print}"
                      f"- {running_average_eval:.2f} avg reward - nodes per level {n_nodes}")
        return self.best_node

    def select(self) -> MCTSNode:
        """ Select a leaf node to start from
        
        Return:
            Node choosen
        
        """
        node = self.root
        path = [self.root]
        actions_ids = []

        while not node.is_leaf:
            # selection_criterion = UCB(
            #     node.children_values, node.children_visits, self.exploration)
            # action_id = np.argmax(selection_criterion)
            action_ids = range(len(node.legal_actions))
            action_id = np.random.choice(action_ids)
            node = node.get_child_by_id(action_id)
            path.append(node)
            actions_ids.append(action_id)

        return node, path, actions_ids

    def simulation(self, start_node:MCTSNode):
        """ Choose a full lineup building nodes on the way
        
        Return:
            Path of nodes used in simulation
            Action IDs used at each node
        
        """

        node = start_node
        path = [start_node]
        actions_ids = []

        while not node.is_terminal:
            action_ids = range(len(node.legal_actions))
            action_id = np.random.choice(action_ids)

            node = node.get_child_by_id(action_id)
            path.append(node)
            actions_ids.append(action_id)

        return path, actions_ids

    def evaluate(self, node, verbose=0) -> float:
        """ Evaluate a completed lineup
        
        Return:
            Value of the node
        
        """
        if self.best_node is None or self.best_node.value < node.value:
            self.best_node = node
            if verbose > 0:
                print(f'New best node: {node} with value {node.value}')
        return node.value

    def backpropagate(self, reward, path, actions_ids):
        """ Backpropagate a reward through a trajectory """
        for node, action_id in zip(path[:-1], actions_ids):
            node.children_visits[action_id] += 1
            node.children_values[action_id] = np.maximum(reward, node.children_values[action_id])
