# !/usr/bin/python

"""
a executer une seule fois 

permet de renommer les noms de répertoire en les numérotant de 1 à 3557
"""

import os

#dataset path
path = "/root/Documents/PLDAC/data/"

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






