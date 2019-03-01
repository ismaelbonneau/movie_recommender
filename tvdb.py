#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 28 17:03:14 2019

@author: ismael Bonneau

Ce fichier sert à récupérer des informations sur l'API the TV db
lien : https://api.thetvdb.com/swagger

Il cherche parmi toutes les séries du dataset celles qui se trouvent sur le site
et construit un fichier CSV contenant plusieurs données:

- nom de la série (tel que dans le dataset)
- id de la série sur le site the TV db
- id de la série sur le site imdb
- id de la série sur le site zap2itId
- genre(s) de la série, comme une chaine de caractères de genres séparés par des tirets, par ex:
  "aventure-drame-fiction"
"""



import re
import requests
import pandas as pd

#essai:
series = ['2733_NCIS__Los_Angeles', '2956_Criminal_Minds', '1041_CSI__Crime_Scene_Investigation', '1910_NCIS',
 '1830_CSI__Miami', '207_Bones', '2212_The_Mentalist', '2767_The_Blacklist', '413_Dexter', '1845_Sherlock_(2010)',
 '884_The_X-Files', '3259_Stargate_SG-1', '381_Star_Trek__The_Next_Generation', '2120_Doctor_Who_(1963)',
 '2091_Star_Trek__Deep_Space_Nine', '384_Twilight_Zone', '186_Doctor_Who', '25_Friends', '1704_Rick_and_Morty',
 '2469_The_Simpsons', '2556_The_Big_Bang_Theory', '292_Modern_Family', '1718_South_Park', '95_How_I_Met_Your_Mother',
 '3012_Grey_s_Anatomy', '2261_Buffy_The_Vampire_Slayer', '175_The_Walking_Dead', '1262_Lost', '1039_Narcos',
 '818_Gomorra_(2014)', '2123_Sense8', '3280_Peaky_Blinders', '121_Suits', '76_Breaking_Bad', '217_Game_of_Thrones',
 '2567_House', '1701_Outlander', '2936_Desperate_Housewives', '2053_Charmed', '345_Dallas', '1641_Pretty_Little_Liars',
 '3314_Shameless', '1906_The_Vampire_Diaries', '196_Smallville']


from utils.load_data import getMostImportantSeries

path = "dataset"

seriesplusimportantes, _ = getMostImportantSeries(path)

lol = list(set(series) | set(seriesplusimportantes[:1000]))
series = lol
series = [serie.replace("__", "_") for serie in series]

#==================================================#
# Authentification
#==================================================#
with open("apikey.auth", 'r') as f:
    apikey = f.readline().rstrip()
    uniqueId = f.readline().rstrip()
    username = f.readline().rstrip()
    
print("apikey={}, username={}, uniqueId={}".format(apikey, username, uniqueId))

auth = {
  "apikey": apikey,
  "userkey": uniqueId,
  "username": username
}

#envoyer une requete post pour récupérer le token d'acces
r = requests.post("https://api.thetvdb.com/login", json=auth)
if r.status_code == 200:
    
    access_token = r.json()['token'] #recup le token d'acces à l'api
    
    print("successfully logged into theTVdb API")
    print("....................................")
    
    #CREATION dataframe contenant les infos récupérées - pour un enregistrement en CSV
    COLUMN_NAMES=['seriesname','id','imdbId','zap2itId','genres']
    dataframe = pd.DataFrame(columns=COLUMN_NAMES)

    for name in series:
        #si le nom de série contient une date, on doit en tenir compte
        name = name.replace("_s__", "s_") #trick pour ne pas louper les grey's anatomy et autre à cause du S de la possession
        name = name.replace("_s_", "s_")
        contientdate = re.search("\(\d\d\d\d\)", name)
        date = None
        if contientdate:
            date = str(contientdate.group(0)).replace("(", "").replace(")", "") #extraction de l'année (année de première sortie)
            parsedname = ("%20".join([mot for mot in name.split("_")[1:-1]])).rstrip()
            truename = " ".join([mot for mot in name.split("_")[1:-1]])
        else:
            parsedname = ("%20".join([mot for mot in name.split("_")[1:]])).rstrip()
            truename = " ".join([mot for mot in name.split("_")[1:]])
        #print("looking for ", truename)
        req = requests.get("https://api.thetvdb.com/search/series", params={"name": parsedname}, headers={'Authorization': "Bearer "+access_token})
        found = False
        ID = ""
        if req.status_code == 200 and "Error" not in req.json():
            response = req.json()['data']
            if len(req.json()) == 1:
                response = req.json()['data'][0]
                if date != None:
                    if response["firstAired"].split("-")[0] == date:
                        #print("found {}, id={}".format(truename, rep["id"]))
                        found = True
                        ID = str(response['id'])
                        seriesname = response["seriesName"]
                else:
                    #print("found {}, id={}".format(truename, rep["id"]))
                    found = True
                    ID = str(response['id'])
                    seriesname = response["seriesName"]
                
            else:
                for rep in response:
                    if rep["seriesName"] == truename or rep["slug"] == "-".join([mot.lower() for mot in parsedname.split("%20")]) or truename in rep["aliases"]:
                        if date != None:
                            if rep["firstAired"].split("-")[0] == date:
                                #print("found {}, id={}".format(truename, rep["id"]))
                                found = True
                                ID = str(rep['id'])
                                seriesname = rep["seriesName"]
                        else:
                            #print("found {}, id={}".format(truename, rep["id"]))
                            found = True
                            ID = str(rep['id'])
                            seriesname = rep["seriesName"]
                    
        if found:
            #2e appel d'api:
            req2 = requests.get("https://api.thetvdb.com/series/"+ID, headers={'Authorization': "Bearer "+access_token})
            if req2.status_code == 200:
                rep2 = req2.json()['data']
                row = [name, str(ID), rep2['imdbId'], rep2['zap2itId'], "-".join(rep2['genre'])]
                dataframe.loc[len(dataframe)] = row #très inefficace
            else:
                print("series API request failed - error {}".format(req2.status_code))
        else:
            print("not found ", name, parsedname, truename, date) #debugging
            
    print("-------------------------------")    
    print("found {} series out of {}".format(len(dataframe), len(series)))
    
    dataframe.to_csv(path_or_buf="series.csv", header=True, encoding="utf-8")
                
else:
    print("authentification impossible... error {}".format(r.status_code))