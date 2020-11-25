from Preprocessing import GetData
import pandas as pd

path = '../Data/preprocessed_data.csv'


def print_draw_predictions_betting_companies():
    """
    function that calculates the precentage of predicted draws from the betting companies
    :return: the percentage of draws
    """
    data = GetData.load_preprocessed_data()
    dra_predictions = 0
    for index, row in data.iterrows():
        if row['avg_draw_odds'] > row['avg_home_win_odds'] and row['avg_draw_odds'] > row['avg_away_win_odds']:
            dra_predictions = dra_predictions + 1

    return dra_predictions / len(data)


def betting_accuracy():
    """
    function that reads all matches from the preprocessed data and calculates the betting companies accuracy
    :return: the accuracy of the betting companies
    """
    data = GetData.load_preprocessed_data()

    correct = 0
    matches = 0

    for index, row in data.iterrows():
        matches = matches + 1
        odds = [row['avg_home_win_odds'], row['avg_draw_odds'], row['avg_away_win_odds']]
        i = odds.index(max(odds))
        result = row['match_result']

        if i == result:
            correct = correct + 1

    return correct / matches


def win_statistics():
    """
    function that reads all matches and calculates the percentage of home wins, draws and away wins
    :return: a list with the percentage of home wins, draws and away wins
    """
    data = GetData.load_preprocessed_data()

    results = [0, 0, 0]
    matches = 0
    for index, row in data.iterrows():
        matches = matches + 1
        results[int(row['match_result'])] = results[int(row['match_result'])] + 1

    return results[0]/matches, results[1]/matches, results[2]/matches


def match_statistics():
    data = GetData.get_all_matches()
    data.describe().to_csv('../Data/Statistics/stats.csv', index=True)
    print(data.describe())


def player_statistics():
    data = GetData.get_all_players()
    data.describe().to_csv('../Data/Statistics/player_stats.csv', index=True)
    print(data.describe())


def match_statistics():
    """
    function that prints the percentage of home wins, draws, away wins and prints the accuracy of the betting companies
    """
    print('Home Wins [%]:', win_statistics()[0]*100)
    print('Draws [%]:', win_statistics()[1]*100)
    print('Away Wins [%]:', win_statistics()[2]*100)
    print('------------------------------------------')
    print('Betting Companies accuracy [%]:', betting_accuracy()*100)
    print('Draws predicted by betting companies [%]:', print_draw_predictions_betting_companies()*100)


def raw_countries_statistics():
    """
    function that reads all rows from Country and prints statistics
    """
    countries = pd.read_sql("SELECT * FROM Country", GetData.conn)

    print(countries.isnull().sum())
    print(countries.describe())

    for index, row in countries.iterrows():
        print(type(row['id']))

    print(countries.dtypes)


def raw_league_statistics():
    """
    function that reads all rows from League and prints statistics
    """
    leagues = pd.read_sql("SELECT * FROM League", GetData.conn)

    print(leagues.isnull().sum())
    print(leagues.describe())
    print(leagues.dtypes)


def raw_match_statistics():
    """
    function that loads all matches and prints statistics
    """
    matches = GetData.get_all_matches()

    print(matches.isnull().sum())
    print('Columns:', len(matches.columns))

    print('rows with nans:', len(matches) - len(matches.dropna()))

    matches.isnull().sum().to_csv('/Users/vegardhaneberg/Desktop/isnullmatches.csv', index=False)
    print(matches.describe())
    print(matches.dtypes)


def raw_player_statistics():
    """
    function that loads all Players and prints statistics
    """
    players = GetData.get_all_players()

    print(players.isnull().sum())
    print('Columns:', len(players.columns))

    print('rows with nans:', len(players) - len(players.dropna()))

    print(players.describe())
    print(players.dtypes)


def raw_player_attributes_statistics():
    """
    function that reads all rows from Player_Attributes and prints statistics
    """
    player_attributes = pd.read_sql("SELECT * FROM Player_Attributes", GetData.conn)

    print('Columns:', len(player_attributes.columns))

    print('rows with nans:', len(player_attributes) - len(player_attributes.dropna()))

    print(player_attributes.describe())
    print(player_attributes.dtypes)


def raw_team_statistics():
    """
    function that reads all rows from Team and prints statistics
    """
    teams = pd.read_sql("SELECT * FROM Team", GetData.conn)

    print('Columns:', len(teams.columns))

    print('rows with nans:', len(teams) - len(teams.dropna()))

    print(teams.describe())
    print(teams.dtypes)

    print('\n\n')
    for col in teams.columns:
        print(col)


def raw_team_attributes_statistics():
    """
    function that reads all rows from Team_Attributes and prints statistics
    """
    team_attributes = pd.read_sql("SELECT * FROM Team_Attributes", GetData.conn)

    print('Columns:', len(team_attributes.columns))

    print('rows with nans:', len(team_attributes) - len(team_attributes.dropna()))

    print(team_attributes.describe())
    print(team_attributes.dtypes)



