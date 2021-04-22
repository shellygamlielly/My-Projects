import csv
from io import TextIOWrapper
from zipfile import ZipFile

MAX_LINES = 10000

# opens file for olympics table.
# CHANGE!
outfile = open("Athlete.csv", 'w', )
outwriterAthlete = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_NONE)
outfile = open("Team.csv", 'w', )
outwriterTeam = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_NONE)
outfile = open("Games.csv", 'w', )
outwriterGames = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_NONE)
outfile = open("Event.csv", 'w', )
outwriterEvent = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_NONE)
outfile = open("Participated.csv", 'w', )
outwriterParticipated = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_NONE)


# process_file goes over all rows in original csv file, and sends each row to process_row()
# DO NOT CHANGE!!!
def process_file():
    counter = 0
    with ZipFile('athlete_events.csv.zip') as zf:
        with zf.open('athlete_events.csv', 'r') as infile:
            reader = csv.reader(TextIOWrapper(infile, 'utf-8'))
            for row in reader:
                # pre-process : remove all quotation marks from input and turns NA into null value ''.
                row = [v.replace(',','') for v in row]
                row = [v.replace("'",'') for v in row]
                row = [v.replace('"','') for v in row]
                row = [v if v != 'NA' else "" for v in row]
                # in 'Sailing', the medal winning rules are different than the rest of olympic games, so they are discarded.
                if row[12] == "Sailing":
                    continue
                # In all years but 1956 summer, olympic games took place in only one city. we clean this fringe case out of the data.
                if row[9] == '1956' and row[11] == 'Stockholm':
                    continue
                # This country is associated with two different noc values, and is discarded.
                if row[6] == 'Union des Socits Franais de Sports Athletiques':
                    continue
                process_row(row)
                counter += 1
                if counter == MAX_LINES:
                    break


# process_row should splits row into the different csv table files
# CHANGE!!!
rows = []
ID = 0
NAME = 1
SEX = 2
AGE = 3
HEIGHT = 4
WEIGHT = 5
TEAM = 6
NOC = 7
GAMES = 8
YEAR = 9
SEASON = 10
CITY = 11
SPORT = 12
EVENT = 13
MEDAL = 14

def process_row(row):
    athleteRow = row[ID:SEX + 1]
    athleteRow.append(row[HEIGHT])
    if str(athleteRow) not in rows:
        rows.append(str(athleteRow))
        outwriterAthlete.writerow(athleteRow)

    teamRow = row[TEAM:NOC + 1]
    if str(teamRow) not in rows:
        rows.append(str(teamRow))
        outwriterTeam.writerow(teamRow)

    gamesRow = row[GAMES:CITY + 1]
    if str(gamesRow) not in rows:
        rows.append(str(gamesRow))
        outwriterGames.writerow(gamesRow)

    eventRow = row[EVENT:SPORT + 1]
    eventRow.append(row[GAMES])
    if str(eventRow) not in rows:
        rows.append(str(eventRow))
        outwriterEvent.writerow(eventRow)

    participatedRow = []
    participatedRow.append(row[ID])
    participatedRow.append(row[TEAM])
    participatedRow.append(row[EVENT])
    participatedRow.append(row[GAMES])
    participatedRow.append(row[AGE])
    participatedRow.append(row[WEIGHT])
    participatedRow.append(row[MEDAL])
    if str(participatedRow) not in rows:
        rows.append(str(participatedRow))
        outwriterParticipated.writerow(participatedRow)


# return the list of all tables
# CHANGE!!!
def get_names():
    return ["Athlete", "Team", "Games", "Event", "Participated"]


if __name__ == "__main__":
    process_file()

