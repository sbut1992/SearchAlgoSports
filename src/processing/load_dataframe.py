import numpy as np
import pandas as pd

def complete_eligibility(dataframe, positions_to_fill):

    def modify_position_eligibility(row):
        try:
            position_list = row.split('/')
            for val in position_list:
                if 'F' in val and 'F' not in position_list:
                    position_list.append('F')
                if 'G' in val and 'G' not in position_list:
                    position_list.append('G')
            position_list.append('UTIL')
            return '/'.join(position_list)
        except:
            position_list = [row]
            if 'F' in row:
                position_list.append('F')
            if 'G' in row:
                position_list.append('G')
            position_list.append('UTIL')
            return '/'.join(position_list)

    dataframe['POSITIONS'] = dataframe['POSITIONS'].apply(modify_position_eligibility)
    dataframe = dataframe.sort_values(by='SALARY',ascending=False)

def load_dataframe(path, positions_to_fill):
    dataframe = pd.read_csv(path)
    complete_eligibility(dataframe, positions_to_fill)
    return dataframe

def get_legal_actions(salaries, names, available_pos, empty_positions,
        players_lineup:list, budget_left:float, level:int):

    players_not_in_lineup = ~np.isin(names, players_lineup)
    try:
        player_is_available = available_pos[:, level]
    except IndexError:
        return []

    player_is_available = np.logical_and(player_is_available, players_not_in_lineup)
    player_is_valid = np.logical_and(player_is_available, salaries <= budget_left)
    legal_actions = np.where(player_is_valid)

    return legal_actions[0]
