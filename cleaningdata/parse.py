# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

#les dossiers nommés "-1" sont vides, on les supprime

import codecs
import srtparser
import os
import glob
import languageClassifier

def cleanData( path ):
    toDelete = []
    for serie in os.listdir(path):
        print(os.path.basename(serie))
        #check the seasons in the serie
        seasons = os.listdir(os.path.join(path,serie))
        #if no season then delete serie directory later
        seasonCount = len(seasons)

        for season in seasons:
            if os.path.isdir(os.path.join(path,serie+os.sep+season)):
                if season == "-1":
                    #supprimer le répertoire -1 qui est toujours vide
                    seasonCount=-1
                    os.rmdir(os.path.join(path,serie+os.sep+season))
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
                            lines = srtparser.parse(episode)
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
        os.rmdir(directory)

                    
