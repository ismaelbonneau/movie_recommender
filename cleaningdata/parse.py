# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

#les dossiers nommés "-1" sont vides, on les supprime

import codecs
import srtparser
import os
import glob

#dataset path
path = "/root/Documents/PLDAC/data/"

for serie in os.listdir(path):
    print(os.path.basename(serie))
    for season in os.listdir(os.path.join(path,serie)):
        if os.path.isdir(os.path.join(path,serie+os.sep+season)):
            if season == "-1":
                #supprimer le répertoire -1 qui est toujours vide
                os.rmdir(os.path.join(path,serie+os.sep+season))
            else:
                print("\tsaison: "+season)
                episodes = glob.glob(os.path.join(path,serie+os.sep+season+os.sep+"*.txt"))
                for episode in episodes:
                    print("\t\t"+os.path.basename(os.path.splitext(episode)[0]))
                    lines = srtparser.parse(episode)
                    lines = "\n".join(lines)
                    newfilename = os.path.basename(os.path.splitext(episode)[0])
                    with codecs.open(os.path.join(path,serie+os.sep+season+os.sep+newfilename+".lines"), "w", "utf-8") as file:
                        file.write(lines)
                    os.remove(episode)

                    
