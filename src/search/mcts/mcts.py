import numpy as np
from copy import copy

from src.processing.load_dataframe import get_legal_actions
from src.search.mcts.selection import UCB

class MCTSNode():

    def __init__(self, dataframe, lineup:list,
                 empty_positions:list, budget_left:float, nodes:list):
        self.dataframe = dataframe
        self.lineup = lineup
        self.lineup.sort()
        self.budget_left = budget_left
        self.empty_positions = empty_positions

        self.nodes = nodes
        self.legal_actions = get_legal_actions(
            dataframe, empty_positions, self.players_lineup, self.budget_left)

        self.children = [None for action in self.legal_actions]
        self.children_visits = np.array([0 for action in self.legal_actions])
        self.children_values = np.array([0 for action in self.legal_actions])
        self.children_cost = np.array([
            self.dataframe[self.dataframe['PLAYER_NAME']==action]['SALARY'].values[0]
            for action in self.legal_actions
        ])

        self.value = None

    def build_children(self, action:str):
        new_empty_positions = copy(self.empty_positions)
        position = self.dataframe[self.dataframe['PLAYER_NAME']==action]['POSITIONS'].values[0]
        if '/' in position:
            positions = position.split('/')
            for position_possibility in positions:
                if position_possibility in new_empty_positions:
                    position = position_possibility
                    break

        new_empty_positions.remove(position)

        new_lineup = self.lineup + [(position, action)]
        new_lineup.sort()
        coresponding_nodes = list(filter(lambda n: n == new_lineup, self.nodes))
        if len(coresponding_nodes) > 0:
            return coresponding_nodes[0]

        cost = self.dataframe[self.dataframe['PLAYER_NAME']==action]['SALARY'].values[0]
        new_budget_left = self.budget_left - cost

        new_node = MCTSNode(self.dataframe, new_lineup, new_empty_positions,
                            new_budget_left, nodes=self.nodes)
        self.nodes.append(new_node)
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
        return [player for _, player in self.lineup]

    @property
    def is_leaf(self) -> bool:
        return len(self.legal_actions) == 0 or np.any(self.children_visits <= 0)

    @property
    def is_terminal(self) -> bool:
        return len(self.legal_actions) == 0 #TODO Or no budget left

    def __repr__(self):
        return f"Node({self.budget_left}$, {self.lineup})"

    def __eq__(self, other):
        if isinstance(other, list):
            return self.lineup == other
        return self.lineup == other.lineup

class MCTSTree():

    def __init__(self, dataframe, empty_positions, budget, exploration=1.):
        self.nodes = []
        self.dataframe = dataframe
        self.root = MCTSNode(dataframe, [], empty_positions, budget, nodes=self.nodes)
        self.nodes.append(self.root)
        self.exploration = exploration
        self.best_node = None

    def run(self, n_simulations=1):
        """ Run the MCTS search for a number of simulations """
        for sim in range(n_simulations):
            start_node, path, actions_ids = self.select()
            sim_path, sim_actions_ids = self.simulation(start_node)
            reward = self.evaluate(sim_path[-1])
            self.backpropagate(reward, path[:-1] + sim_path, actions_ids + sim_actions_ids)
            if sim == 0 or (sim+1)%(n_simulations//10)==0:
                print(f"{sim+1}/{n_simulations} - {len(self.nodes)} nodes")

    def select(self) -> MCTSNode:
        """ Select a leaf node to start from
        
        Return:
            Node choosen
        
        """
        node = self.root
        path = [self.root]
        actions_ids = []

        while not node.is_leaf:
            selection_criterion = UCB(
                node.children_values, node.children_visits, self.exploration)
            best_child_id = np.argmax(selection_criterion)
            node = node.get_child_by_id(best_child_id)
            path.append(node)
            actions_ids.append(best_child_id)

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

    def evaluate(self, node) -> float:
        """ Evaluate a completed lineup
        
        Return:
            Value of the node
        
        """
        if not node.is_terminal:
            raise ValueError("Non-terminal nodes should never be evaluated")
        if node.value is None:
            node.value = self.dataframe[
                self.dataframe['PLAYER_NAME'].isin(node.players_lineup)
            ]['FPTS'].sum()

        if self.best_node is None or self.best_node.value < node.value:
            self.best_node = node
            print(f'New best node: {node} with value {node.value}')

        return node.value

    def backpropagate(self, reward, path, actions_ids):
        """ Backpropagate a reward through a trajectory """
        for node, action_id in zip(path[:-1], actions_ids):
            node.children_visits[action_id] += 1
            n = node.children_visits[action_id]
            q = node.children_values[action_id]
            node.children_values[action_id] += (reward - q) / n
