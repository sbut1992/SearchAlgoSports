import numpy as np

def UCB(values, visits, c=np.sqrt(0.5), unexplored_value=np.inf):
    exploration = c * np.sqrt( np.log(np.sum(visits)) / visits )
    exploration = np.where(visits == 0, unexplored_value, exploration)
    return values + exploration

def UCB_tuned(values, visits, c=np.sqrt(2), unexplored_value=np.inf):
    second_term = min(1/4, (np.var(values) + (2*np.log(np.sum(visits))/visits)))
    exploration = np.sqrt( (np.log(np.sum(visits)) / visits) * second_term)
    # exploration = c * np.sqrt( np.log(np.sum(visits)) / visits )
    exploration = np.where(visits == 0, unexplored_value, exploration)
    return values + exploration
