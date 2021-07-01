import numpy as np
import pandas as pd

def complete_eligibility(dataframe, positions_to_fill = ['PG','SG','SF','PF','C','F','G','UTIL']):

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


def load_dataframe(path):
    dataframe = pd.read_csv(path)
    complete_eligibility(dataframe)
    return dataframe

def get_legal_actions(dataframe, empty_positions, players_lineup, budget_left):

    def filter_available(row):
        position_list = row.split('/')
        for val in position_list:
            if val in empty_positions:
                return True
        return False

    player_is_available = dataframe['POSITIONS'].apply(filter_available)
    players_not_in_lineup = np.logical_not(dataframe['PLAYER_NAME'].isin(players_lineup))
    budget_is_enough = np.array(dataframe['SALARY']) <= budget_left
    player_is_valid = np.logical_and(player_is_available, players_not_in_lineup, budget_is_enough)
    return list(dataframe[player_is_valid]['PLAYER_NAME'].unique())

