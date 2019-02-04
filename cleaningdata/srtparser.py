# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

#fichier servant à nettoyer un fichier .srt 
#et récupérer les lignes de dialogue le composant.

import codecs
import re

url_regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
timestamp_regex = r'(2[0-3]|[01][0-9]|[0-9]):([0-5][0-9]|[0-9]):([0-5][0-9]|[0-9])'
startingTag_regex = r'<[uib]+>'
closingTag_regex = r'</[uib]+>'
uselessTag_regex = r'<[a-zA-Z]+ .*>'

#les fichiers .srt peuvent contenir des balises qui encadrent du texte:
# - gras : <b>...</b> 
# - italique : <i>...</i> 
# - souligné : <u>...</u>

#d'autres balises comme des balises de font (police d'écriture) peuvent aussi être présentes
#il faut également se débarrasser des crédits (translated by ***, uploaded by ***, etc)

def containsUselessURL(line):
	return bool(re.search(uselessTag_regex, line))

def containsURL(line):
	return bool(re.search(url_regex, line)) or bool(re.search(r'www', line))

def startingTag(line):
	return bool(re.search(startingTag_regex, line))

def closingTag(line):
	return bool(re.search(closingTag_regex, line))

def islowercase(char):
	return (char.lower() == char and char.isalpha())

def firstPerson(line):
	"""
	renvoie True si la ligne commence par un I
	-le seul cas en anglais où une majuscule ne commence par la phrase-
	suppose une phrase en anglais.
	"""
	if len(line) == 1:
		if line[0] == "I":
			return True
		else: 
			return False

	if line[0] == "I" and line[1] in " '":
		return True
	return False


def timestamp(line):
	"""
	renvoie True si la ligne contient un timestamp
	"""
	return bool(re.match(timestamp_regex, line))

def istext(line):
  if re.search('[a-zA-Z]', line):
    return True
  return False


def parse(filename):
	"""
	parse un fichier .srt pour ne garder que les lignes de dialogue.
	"""

	#A COMPLETER

	with codecs.open(filename, "r", 'utf-8', errors='ignore') as file:
		strings = file.readlines()

		newlines = []
		i = 0
		while i < len(strings):
			line = strings[i]
			#ne garder que les lignes de texte
			if not(timestamp(line)) and (istext(line)):
				text = line.rstrip()

				if not(containsURL(text)) and not(containsUselessURL(text)):

					text = re.sub(startingTag_regex,'', text)
					text = re.sub(closingTag_regex,'', text)
					if text != "":

						if len(newlines) != 0:
							#dans ces cas on considère comme faisant partie de la ligne précédente:
							if islowercase(text[0]) or (text[0] == ','):
								newlines[-1] = newlines[-1] + " " + text
							elif firstPerson(text) and newlines[-1][-1] not in ".?":
								newlines[-1] = newlines[-1] + " " + text
							else:
								newlines.append(text)
						else:
							newlines.append(text)
			i += 1

		return newlines


