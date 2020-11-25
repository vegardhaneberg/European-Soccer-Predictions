import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import plotly.express as px

from Preprocessing import GetData

path = "../Data/"
database = path + 'database.sqlite'

conn = sqlite3.connect(database)


def preprocessing():
    """
    function that loads the data from the database and processes it so that it is possible to plot features against the
    match result. Helps discover important features to include in the models.
    :return: a dataframe containing the number of times the home team beat the away team in their last four encounters,
    the number of times the away team beat the home team in their last four encounters and the match result
    """
    print('Starting preprocessing of data...')

    matches = pd.read_sql("SELECT * FROM Match;", conn)

    input_vector = []

    for index, match in matches.iterrows():
        if index % 100 == 0:
            print('Processed', index, 'of', len(matches), 'samples...')

        encounters = get_last_results(matches, match['home_team_api_id'], match['away_team_api_id'], match['date'])
        if encounters is not False:
            home_team_encounter_wins = encounters[0]
            away_team_encounter_wins = encounters[2]
            match_result = get_match_result(match['home_team_goal'], match['away_team_goal'])

            input_vector.append([home_team_encounter_wins, away_team_encounter_wins, match_result])

    return pd.DataFrame(input_vector, columns=['home_team_encounter_wins', 'away_team_encounter_wins', 'match_result'])


def get_match_result(home, away):
    """
    functoin that creates the match result as 1 if home team is larger than away team, 1 if draw and 2 if away win.
    :param home: goals scored by the home team
    :param away: goals scored by the away team
    :return: 0 for home win, 1 for draw and 2 for away win
    """
    if home > away:
        return 0
    elif home == away:
        return 1
    else:
        return 2


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

    if games == 0:
        return False

    win_percentage_vector = [result_vector[0], result_vector[1], result_vector[2]]

    return win_percentage_vector


def plot_encounters():
    """
    function that plots the number of times the home team beat the away team against the mattch results.
    """

    data = preprocessing()
    data['match_result'] = data['match_result'].apply(lambda x: 'Home Win' if x == 0 else x)
    data['match_result'] = data['match_result'].apply(lambda x: 'Draw' if x == 1 else x)
    data['match_result'] = data['match_result'].apply(lambda x: 'Away Win' if x == 2 else x)

    fig = px.histogram(data, x='home_team_encounter_wins', color='match_result',
                       hover_data=data.columns,
                       template='plotly_white',
                       labels={'match_result': 'Match Result'},
                       category_orders={'match_result': ['Home Win', 'Draw', 'Away Win']},
                       color_discrete_map={'Home Win': 'peachpuff', 'Draw': 'thistle',
                                           'Away Win': 'rebeccapurple'},
                       barnorm='percent')
    fig.update_layout(yaxis_title_text='Percentage',
                      xaxis_title_text='Number of Times the Home Team Beat the Away Team',
                      bargap=0.2,
                      font=dict(
                          family='Courier New, monospace',
                          size=35
                      ))
    fig.show()


def pair_plot_ratings():
    """
    function that plots the two teams player rating against the match result
    """

    data = GetData.load_preprocessed_data()

    data['match_result'] = data['match_result'].apply(lambda x: 'Home Win' if x == 0 else x)
    data['match_result'] = data['match_result'].apply(lambda x: 'Draw' if x == 1 else x)
    data['match_result'] = data['match_result'].apply(lambda x: 'Away Win' if x == 2 else x)

    data = data[(data['avg_home_rating'] != 0)]
    data = data[(data['avg_away_rating'] != 0)]
    sns.pairplot(data=data[['avg_home_rating', 'avg_away_rating', 'match_result']].sample(5000), hue="match_result",
                 palette={'Home Win': 'peachpuff', 'Draw': 'thistle',
                          'Away Win': 'rebeccapurple'}
                 )

    plt.show()


pair_plot_ratings()
plot_encounters()
