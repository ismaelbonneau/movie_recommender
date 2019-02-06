#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:30:12 2019

@author: Ismael Bonneau & Issam Benamara
"""
import codecs
import os
import sys
import glob
import re
import shutil
sys.path.append(os.path.abspath('../classifiers'))
import languageClassifier

def cleanData():
    """ External dataset Path """
    
    with open('dataPath.txt') as f:
        lines = f.readlines()
        path = lines[0].strip('\n')
    
    
    """ parsing/cleaning files """
    # extract only meaningful ( linguistic ) text from all the data
    # removing empty directories
    # remove non english episodes
    
    cleanOnly(path)
    rename(path)



def cleanOnly( path ):
    """ 
    output :
        - no useless files ( grab and info )
        - no empty folders or files
        - no NONenglish episodes
        - parsed srt files to linguistic only text files ( one life of dialogue per line )
    """
    toDelete = []
    walk = next(os.walk(path))
    series = walk[1]
    
    for serie in series:
        print(os.path.basename(serie))
        #check the seasons in the serie
        serieWalk = next(os.walk(os.path.join(path,serie)))
        seasons = serieWalk[1]
        
        for f in serieWalk[2]:
            fp = os.path.join(os.path.join(path,serie),f)
            os.remove(fp)
            print( "Removed : " + fp )
            
        #if no season then delete serie directory later
        seasonCount = len(seasons)

        for season in seasons:
            if season == "-1":
                #supprimer le répertoire -1 qui est toujours vide
                seasonCount=-1
                toDelete.append(os.path.join(path,serie+os.sep+season))
            else:
                print("\tsaison: "+season)
                # check episodes of season
                episodes = glob.glob(os.path.join(path,serie+os.sep+season+os.sep+"*.txt"))
                #if no episodes then delete season directory later
                episodesCount = len(episodes)
                
                for episode in episodes:
                    #check if empty episode
                    if( len(episode) != 0 ):
                        print("\t\t"+os.path.basename(os.path.splitext(episode)[0]))
                        lines = parse(episode)
                        lines = "\n".join(lines)
                        
                        if languageClassifier.languageDetector(lines) == "english":
                            newfilename = os.path.basename(os.path.splitext(episode)[0])
                            with codecs.open(os.path.join(path,serie+os.sep+season+os.sep+newfilename+".lines"), "w", "utf-8") as file:
                                file.write(lines)
                        else:
                            episodesCount-=1
                    else:
                        episodesCount-=1
                    os.remove(episode)
                    
                if episodesCount == 0 :
                    toDelete.append(os.path.join(path,serie+os.sep+season))
                    seasonCount-=1
                        
        if seasonCount == 0:
            toDelete.append(os.path.join(path,serie))
            
    for directory in toDelete:
        shutil.rmtree(directory)
        print( "Removed : " + directory )
    
    for f in walk[2]:
        fp = os.path.join(path,f)
        os.remove(fp)
        print( "Removed : " + fp )

def rename(path):
    """ naming normalizing """
    print("searching for directories to rename...")
    
    repnames = os.listdir(path)
    i = 1
    for repname in repnames:
    	name = repname.split("___")
    	if len(name) != 2:
    		title = "_".join(name[1:])
    	else:
    		title = name[1]
    
    	#repository name like: index_title_of_the_serie
    	titre = str(i)+"_"+title
    	os.rename(os.path.join(path, repname), os.path.join(path, titre))
    	i+=1
    print("done.")

                    



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


""" THE CALL """
cleanData()