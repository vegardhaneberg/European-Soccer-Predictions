import sqlite3
import pandas as pd
import time
import warnings
import numpy as np
import math as m

warnings.simplefilter("ignore")

try:
    path = "Data/"
    database = path + 'database.sqlite'

    conn = sqlite3.connect(database)
except:
    path = "../Data/"
    database = path + 'database.sqlite'

    conn = sqlite3.connect(database)


def get_all_matches():
    """
    function that loads all matches into a dataframe and prints the time it took to read all matches
    :return: dataframe with all matches
    """
    start = time.time()
    matches = pd.read_sql("SELECT * FROM Match;", conn)
    print('Time to get all matches', time.time() - start)
    return matches


def get_all_players():
    """
    function that loads all players into a dataframe and prints the time it took to read all matches
    :return: dataframe with all players
    """
    start = time.time()
    players = pd.read_sql("SELECT * FROM Player_Attributes", conn)
    print('Time to get all players:', time.time() - start)
    return players


def get_last_matches(matches, team_api_id, date, x=5):
    """
    function that returns the last x matches played by the team with team_api_id before the given date
    :param matches: all matches as a dataframe
    :param team_api_id: id as int for the given team
    :param date: date as string for the given match
    :param x: number of matches to be evaluated
    :return: the last x matches
    """

    team_matches = matches[(matches['home_team_api_id'] == team_api_id) | (matches['away_team_api_id'] == team_api_id)]

    last_matches = team_matches[team_matches.date < str(date)].sort_values(by='date', ascending=False).iloc[0:x, :]

    return last_matches


def get_last_results(matches, first_team_api_id, last_team_api_id, date, x=2):
    """
    function that calculates the average wins, draws and losses for the first_team against the last_team
    :param matches: all matches as a dataframe
    :param first_team_api_id: id as int for the first team
    :param last_team_api_id: id as int for the second team
    :param date: date as string for the given match
    :param x: number of matches to be evaluated
    :return: a list containing the average wins, draws and losses in the last x matches between first team and last team
    """

    first_team_home_encounters = matches[(matches['home_team_api_id'] == first_team_api_id) & (matches['away_team_api_id'] == last_team_api_id)]
    first_team_away_encounters = matches[(matches['home_team_api_id'] == last_team_api_id) & (matches['away_team_api_id'] == first_team_api_id)]

    last_first_team_home_encounters = first_team_home_encounters[first_team_home_encounters.date < str(date)].sort_values(by='date', ascending=False).iloc[0:x, :]
    last_first_team_away_encounters = first_team_away_encounters[first_team_away_encounters.date < str(date)].sort_values(by='date', ascending=False).iloc[0:x, :]

    first_team_home_wins = len(last_first_team_home_encounters[(last_first_team_home_encounters['home_team_goal'] > last_first_team_home_encounters['away_team_goal'])])
    last_team_away_wins = len(last_first_team_home_encounters[(last_first_team_home_encounters['home_team_goal'] < last_first_team_home_encounters['away_team_goal'])])

    last_team_home_wins = len(last_first_team_away_encounters[(last_first_team_away_encounters['home_team_goal'] > last_first_team_away_encounters['away_team_goal'])])
    first_team_away_wins = len(last_first_team_away_encounters[(last_first_team_away_encounters['home_team_goal'] < last_first_team_away_encounters['away_team_goal'])])

    first_team_wins = first_team_home_wins + first_team_away_wins
    last_team_wins = last_team_away_wins + last_team_home_wins
    draws = 2*x - first_team_wins - last_team_wins

    result_vector = [first_team_wins, draws, last_team_wins]
    games = sum(result_vector)
    win_percentage_vector = [result_vector[0] / games, result_vector[1] / games, result_vector[2] / games]

    return win_percentage_vector


def get_last_matches_avg_goals_scored(all_matches, team_api_id, date, x=5):
    """
    function that calculates the average scored goals over the last x matches
    :param all_matches: all matches as a dataframe
    :param team_api_id: id as int for the given team
    :param date: date as string for the given match
    :param x: number of matches to be evaluated
    :return: average goals scored in the x last matches
    """

    last_matches = get_last_matches(all_matches, team_api_id, date, x)
    home_matches = last_matches[(last_matches['home_team_api_id'] == team_api_id)]
    away_matches = last_matches[(last_matches['away_team_api_id'] == team_api_id)]

    if len(last_matches) == 0:
        return 0

    return (home_matches['home_team_goal'].sum(axis=0) + away_matches['away_team_goal'].sum(axis=0)) / \
           (len(last_matches))


def get_last_matches_avg_goals_conceded(all_matches, team_api_id, date, x=5):
    """
    function that calculates the average conceded goals over the last x matches
    :param all_matches: all matches as a dataframe
    :param team_api_id: id as int for the given team
    :param date: date as string for the given match
    :param x: number of matches to be evaluated
    :return: average goals conceded in the x last matches
    """

    last_matches = get_last_matches(all_matches, team_api_id, date, x)
    home_matches = last_matches[(last_matches['home_team_api_id'] == team_api_id)]
    away_matches = last_matches[(last_matches['away_team_api_id'] == team_api_id)]

    if len(last_matches) == 0:
        return 0

    return (home_matches['away_team_goal'].sum(axis=0) + away_matches['home_team_goal'].sum(axis=0)) / \
           (len(last_matches))


def get_avg_last_matches_wins(all_matches, team_api_id, date, x=5):
    """
    function that calculates the average wins in the last x matches
    :param all_matches: all matches as a dataframe
    :param team_api_id: id as int for the given team
    :param date: date as string for the given match
    :param x: number of matches to be evaluated
    :return: average wins in the last x matches
    """

    last_matches = get_last_matches(all_matches, team_api_id, date, x)

    home_matches = last_matches[(last_matches['home_team_api_id'] == team_api_id)]
    away_matches = last_matches[(last_matches['away_team_api_id'] == team_api_id)]

    home_wins = home_matches[(last_matches['home_team_goal'] > last_matches['away_team_goal'])]
    away_wins = away_matches[(last_matches['home_team_goal'] < last_matches['away_team_goal'])]

    if len(last_matches) == 0:
        return 0

    return (len(home_wins) + len(away_wins)) / len(last_matches)


def get_match_result_array(matches):
    """
    function that creates a list with list of match results on the format for neural networks
    :param matches: all matches as a dataframe
    :return: a list with lists with match results on the format for neural networks
    """
    matches_outcome_array = []

    home_goals = matches['home_team_goal'].to_numpy()
    away_goals = matches['away_team_goal'].to_numpy()

    for pos in range(len(home_goals)):

        if home_goals[pos] > away_goals[pos]:
            matches_outcome_array.append([1, 0, 0])
        elif home_goals[pos] < away_goals[pos]:
            matches_outcome_array.append([0, 0, 1])
        else:
            matches_outcome_array.append([0, 1, 0])

    return matches_outcome_array


def get_home_and_away_team_overall_player_rating_avg(match, all_players):
    """
    function that calculates the average player rating for both teams of a match
    :param match: one row from the match table
    :param all_players: all players in the database
    :return: list with average player ratings for the two teams
    """
    match_date = match['date']
    home_player_ids = np.array(match[['home_player_1',
                                      'home_player_2',
                                      'home_player_3',
                                      'home_player_4',
                                      'home_player_5',
                                      'home_player_6',
                                      'home_player_7',
                                      'home_player_8',
                                      'home_player_9',
                                      'home_player_10',
                                      'home_player_11']])
    away_player_ids = np.array(match[['away_player_1',
                                      'away_player_2',
                                      'away_player_3',
                                      'away_player_4',
                                      'away_player_5',
                                      'away_player_6',
                                      'away_player_7',
                                      'away_player_8',
                                      'away_player_9',
                                      'away_player_10',
                                      'away_player_11']])

    home_player_ratings = []
    away_player_ratings = []
    for home_player_id in home_player_ids:
        if np.isnan(home_player_id):
            continue
        home_player = all_players[(all_players['player_api_id'] == home_player_id)]
        if home_player.empty:
            continue
        try:
            home_player_last_match = \
                home_player[home_player.date <= match_date].sort_values(by='date', ascending=False).iloc[0]
            home_player_ratings.append(home_player_last_match['overall_rating'])
        except IndexError:
            print("hjemme")
            home_player.sort_values(by='data', ascending=True)
            temp_index = 0
            bad_index = True
            while bad_index:
                if m.isnan(home_player[temp_index]['overall_rating']):
                    temp_index = temp_index + 1
                else:
                    bad_index = False
            home_player_ratings.append(home_player[temp_index]['overall_rating'])

    for away_player_id in away_player_ids:
        if np.isnan(away_player_id):
            continue
        away_player = all_players[(all_players['player_api_id'] == away_player_id)]
        if away_player.empty:
            continue
        try:
            away_player_last_match = \
                away_player[away_player.date <= match_date].sort_values(by='date', ascending=False).iloc[0]
            away_player_ratings.append(away_player_last_match['overall_rating'])
        except IndexError:
            print("borte")
            away_player.sort_values(by='date', ascending=True)
            temp_index = 0
            bad_index = True
            while bad_index:
                if m.isnan(away_player[temp_index]['overall_rating']):
                    temp_index = temp_index + 1
                else:
                    bad_index = False
            away_player_ratings.append(away_player[temp_index]['overall_rating'])

    if len(home_player_ratings) == 0 or len(away_player_ratings) == 0:
        return [0, 0]
    return [np.mean(home_player_ratings)/100, np.mean(away_player_ratings)/100]


def get_avg_betting_odds(match):
    """
    function that calculates the average betting probabilities for the given match
    :param match: one row in the Match table
    :return: probability for home win, draw and away win for the given match
    """
    home_win_odds = []
    draw_odds = []
    away_win_odds = []
    if not m.isnan(match['B365H']):
        home_win_odds.append(match['B365H'])
        draw_odds.append(match['B365D'])
        away_win_odds.append(match['B365A'])
    if not m.isnan(match['BWH']):
        home_win_odds.append(match['BWH'])
        draw_odds.append(match['BWD'])
        away_win_odds.append(match['BWA'])
    if not m.isnan(match['IWH']):
        home_win_odds.append(match['IWH'])
        draw_odds.append(match['IWD'])
        away_win_odds.append(match['IWA'])
    if not m.isnan(match['LBH']):
        home_win_odds.append(match['LBH'])
        draw_odds.append(match['LBD'])
        away_win_odds.append(match['LBA'])
    if not m.isnan(match['PSH']):
        home_win_odds.append(match['PSH'])
        draw_odds.append(match['PSD'])
        away_win_odds.append(match['PSA'])
    if not m.isnan(match['WHH']):
        home_win_odds.append(match['WHH'])
        draw_odds.append(match['WHD'])
        away_win_odds.append(match['WHA'])
    if not m.isnan(match['SJH']):
        home_win_odds.append(match['SJH'])
        draw_odds.append(match['SJD'])
        away_win_odds.append(match['SJA'])
    if not m.isnan(match['VCH']):
        home_win_odds.append(match['VCH'])
        draw_odds.append(match['VCD'])
        away_win_odds.append(match['VCA'])
    if not m.isnan(match['GBH']):
        home_win_odds.append(match['GBH'])
        draw_odds.append(match['GBD'])
        away_win_odds.append(match['GBA'])
    if not m.isnan(match['BSH']):
        home_win_odds.append(match['BSH'])
        draw_odds.append(match['BSD'])
        away_win_odds.append(match['BSA'])

    if not home_win_odds:
        return 0, 0, 0
    avg_home_win_odds = np.mean(home_win_odds)
    avg_draw_odds = np.mean(draw_odds)
    avg_away_win_odds = np.mean(away_win_odds)
    return 1/avg_home_win_odds, 1/avg_draw_odds, 1/avg_away_win_odds


def store_preprocessed_data(lists, filename):
    """
    function that stores the lists in a csv file. Used to store the preprocessed data
    :param lists: preprocessed data, two lists with lists
    :param filename: filename of the csv file
    """
    df = create_df_from_two_lists(lists)

    df.to_csv('Data/' + filename, index=False)


def create_df_from_two_lists(lists):
    """
    functino that creates a dataframe from two lists
    :param lists: a list containing two lists with the preprocessed data
    :return: a dataframe with the same data as the two lists
    """
    column_names = ['avg_last_home_team_wins',
                   'avg_last_away_team_wins', 'home_team_scored_goals', 'away_team_scored_goals',
                   'home_team_conceded_goals', 'away_team_conceded_goals', 'home_team_encounter_wins',
                   'away_team_encounter_wins', 'avg_home_rating', 'avg_away_rating', 'match_result']

    first_list = lists[0]
    second_list = lists[1]

    if len(first_list) != len(second_list):
        raise Exception('Lengths of lists must be equal in create_df_from_two_lists')

    for pos in range(len(first_list)):
        first_list[pos].append(second_list[pos])

    return pd.DataFrame(first_list, columns=column_names)


def convert_output_vector(convert_list):
    """
    function that converts the output vector from the format of 0, 1 or 2 to a binary vector
    :param convert_list: the match results on the format of 0, 1 or 2
    :return: a list with binary values
    """

    output = []
    for result in convert_list:
        output.append(result.index(max(result)))

    return output


def convert_df_to_lists(df, convert_to_vector=True):
    """
    function that converts a dataframe to two lists
    :param df: input dataframe
    :param convert_to_vector: bolean that decides if the match results are convertet to a vector of 0s or 1s. If true
    the match results are represented as a vector like thiw [0, 1, 0] if false the match results are on the format as
    0, 1 or 2
    :return: a list containing the information in the dataframe
    """

    lists = df.values.tolist()
    input = []
    output = []
    for list in lists:
        input.append(list[:len(list) - 1])
        output.append(int(list[len(list) - 1]))

    if convert_to_vector:
        for pos in range(len(output)):
            match_result = [0, 0, 0]
            match_result[output[pos]] = 1
            output[pos] = match_result

    return input, output


def load_preprocessed_data():
    """
    functino that loads the preprocessed data from a csv file. The function assumes that the csv file is placed in the
    Data folder and is called preprocessed_data.csv
    :return: dataframe with the preprocessed data
    """
    try:
        return pd.read_csv('Data/preprocessed_data.csv', sep=',')
    except:
        return pd.read_csv('../Data/preprocessed_data.csv', sep=',')


def preprocessing(samples=0):
    """
    the main function that preprocesses all data in the database and returns a list with two lists. the first list is
    the input vector to the models, and the second is a list containing the match results
    :param samples: the number of samples to include in the models. Can be set low for faster runtime
    :return: a list with two lists. the first list is the input vector to the models, and the second is a list
    containing the match results
    """
    print('Starting preprocessing of data...')
    start = time.time()
    matches = pd.read_sql("SELECT * FROM Match;", conn)
    all_players = pd.read_sql("""SELECT * FROM Player_Attributes;""", conn)

    passing_time = time.time()
    print('Time to get all matches and players', passing_time - start)
    number_of_removed_samples = 400
    matches = matches.iloc[number_of_removed_samples:]

    if samples != 0:
        matches = matches.head(samples)


    input_vector = []
    output_vector = get_match_results(matches)

    for index, match in matches.iterrows():
        if index % 100 == 0:
            print('Processed', index - number_of_removed_samples, 'of', len(matches), 'samples...')

        avg_home_rating, avg_away_rating = get_home_and_away_team_overall_player_rating_avg(match, all_players)

        avg_home_win_odds, avg_draw_odds, avg_away_win_odds = get_avg_betting_odds(match)

        avg_last_home_team_wins = get_avg_last_matches_wins(matches, match['home_team_api_id'], match['date'], x=7)
        avg_last_away_team_wins = get_avg_last_matches_wins(matches, match['away_team_api_id'], match['date'], x=7)

        home_team_scored_goals = get_last_matches_avg_goals_scored(matches, match['home_team_api_id'], match['date'], x=7)
        away_team_scored_goals = get_last_matches_avg_goals_scored(matches, match['away_team_api_id'], match['date'], x=7)

        home_team_conceded_goals = get_last_matches_avg_goals_conceded(matches, match['home_team_api_id'], match['date'], x=7)
        away_team_conceded_goals = get_last_matches_avg_goals_conceded(matches, match['away_team_api_id'], match['date'], x=7)

        encounters = get_last_results(matches, match['home_team_api_id'], match['away_team_api_id'], match['date'])
        home_team_encounter_wins = encounters[0]
        away_team_encounter_wins = encounters[2]

        input_vector.append([avg_home_win_odds, avg_draw_odds, avg_away_win_odds, avg_last_home_team_wins,
                             avg_last_away_team_wins, home_team_scored_goals, away_team_scored_goals,
                             home_team_conceded_goals, away_team_conceded_goals, home_team_encounter_wins,
                             away_team_encounter_wins, avg_home_rating, avg_away_rating])

    print('Normalizing goals scored and goals conceded...')
    normalize_goals(input_vector, 5)
    normalize_goals(input_vector, 6)
    normalize_goals(input_vector, 7)
    normalize_goals(input_vector, 8)

    print('Time to process the data:', time.time() - passing_time)

    return [input_vector, output_vector]


def get_match_results(matches):
    """
    function that returns the results of all matches as 0 for home win, 1 for draw and 2 for away win
    :param matches: all matches in the database
    :return: a list containing the results of all matches as 0 for home win, 1 for draw and 2 for away win
    """
    matches_outcome_array = []

    home_goals = matches['home_team_goal'].to_numpy()
    away_goals = matches['away_team_goal'].to_numpy()

    for pos in range(len(home_goals)):

        if home_goals[pos] > away_goals[pos]:
            matches_outcome_array.append(0)
        elif home_goals[pos] < away_goals[pos]:
            matches_outcome_array.append(2)
        else:
            matches_outcome_array.append(1)

    return matches_outcome_array


def normalize_goals(input_list, index):
    """
    min max normalization. The changes are made inplace
    :param input_list: a list with lists, where each inner list is a row of preprocessed data
    :param index: the position in the inner lists that will be normalized.
    """
    max_goals = -1
    min_goals = 100000

    for features in input_list:
        if features[index] > max_goals:
            max_goals = features[index]
        if features[index] < min_goals:
            min_goals = features[index]

    for features in input_list:
        features[index] = normalize(features[index], max_goals, min_goals)


def normalize(x, max_number, min_number):
    """
    function that normalized the number x
    :param x: number that will be normalized
    :param max_number: highest number
    :param min_number: lowest number
    :return: the normalized value of x
    """
    return (x - min_number) / (max_number - min_number)


def convert_output_vector_to_nn_format(vector):
    """
    functoin that converts the match results from the format 0, 1, or 2 to a list with 1 on the index of the result
    :param vector: the match results
    :return: a list with lists for each match result on the format for neural networks.
    """

    converted_vector = []

    for result in vector:
        match_result = [0, 0, 0]
        match_result[result] = 1
        converted_vector.append(match_result)

    return converted_vector




