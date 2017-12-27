from bs4 import BeautifulSoup
import requests
import re
import csv
import sys

def getPlayers(url):
	'''
	Get top players list
	'''
	source_code = requests.get(url);
	plain_text = source_code.text;
	soup = BeautifulSoup(plain_text, 'html.parser');
	table = soup.find('table', attrs = {'class':"table table-condensed"})
	PlayerSet = []
	for line in table.find_all('a'):
		if 'protected' not in str(line):
			nameTmp = re.search('.*"/profile/(.*)".*', str(line))
			if nameTmp:
				name = str(nameTmp.group(1));
				print name
				PlayerSet.append(name);

	return PlayerSet

def getMatch(url, player):
	'''
	Collect battle record for each player
	'''
	newURL = url+str(player);
	source_code = requests.get(newURL);
	plain_text = source_code.text;
	soup = BeautifulSoup(plain_text, 'html.parser');
	allDiv = soup.find_all('div', attrs = {'class':"panel panel-inverse"});
	cardSet = [];
	scoreSet = [];
	if len(allDiv)<3:
		return cardSet, scoreSet;
	for match in allDiv[2].find_all('div', attrs = {'class':"panel panel-inverse"}):
		timeLine = str(match.find('div', attrs = {'class':'panel-footer supercell'}).contents);
		#if 'day' in timeLine or 'days' in timeLine:
		#	continue
	
		scoreLine = str(match.find('span', attrs = {'style':'background-color: #000;'}).contents);
		score = str(re.search('.*(. - .).*', scoreLine).group(1))
		cardsTable = match.find_all('ul', attrs = {'class':'deck'});
		if len(cardsTable)>2:
			continue;
		cards = [];
		for cardLine in cardsTable:
			for card in cardLine.find_all('li', attrs = {'class':'spell'}):
				cardURL = str(card.find('img')['src']);
				cards.append(str(re.search('/images/cards/(.*).png', cardURL).group(1)));
		scoreSet.append(score);
		cardSet.append(cards);

	print player, len(cardSet)
	return scoreSet, cardSet;

def readPlayers(filename):
	'''
	Save player list to the file
	'''
	file = open(filename, 'r');
	nameList = []
	for line in file.readlines():
		name = re.match('finish write (.*)', line);
		if name!=None:
			name = name.group(1);
			nameList.append(name);

	return nameList


def main(argv):
	PlayerSet = getPlayers('https://statsroyale.com/top/players/');
	#PlayerSet = readPlayers('playerList417.txt');
	print len(PlayerSet)
	file = open(str(argv), 'w');
	wr = csv.writer(file, dialect='excel');
	for player in PlayerSet:
		scoreSet, cardSet = getMatch('https://statsroyale.com/profile/', player);
		row = [];
		for idx in range(len(scoreSet)):
			row = cardSet[idx];
			score = scoreSet[idx].split(' ');
			row.append(score[0]);
			row.append(score[2]);
			wr.writerow(row)

		print 'finish write ' + player;


if __name__ == "__main__":
    main(sys.argv[1]);
