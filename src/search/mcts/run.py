from src.processing.load_dataframe import load_dataframe
from src.search.mcts.mcts import MCTSTree

if __name__ == '__main__':
    dataframe = load_dataframe('2021_03_12_slate.csv')
    empty_positions = ['PG','SG','SF','PF','C','F','G','UTIL']
    budget = 50000
    mcts_tree = MCTSTree(dataframe, empty_positions, budget, exploration=1.)
    mcts_tree.run(1000)

