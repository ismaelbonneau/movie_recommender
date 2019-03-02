# !/usr/bin/python
# -*- coding: utf-8 -

#=========================================
# Scrapping du site imdb afin de récupérer
# des couples rating - user pour différentes séries
#=========================================


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup

# Utilisation d'une version de chrome automatisée 
# Methode nécessaire à la récupération de TOUS les avis sur la page

#options pour chrome
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu");
options.add_argument("--disable-extensions");
options.add_experimental_option("useAutomationExtension", False);
#options.add_argument("--start-maximized");
#options.add_argument("--headless") #cacher la fenêtre

#créer un nouveau robot chrome
driver = webdriver.Chrome("/home/ismael/Documents/chromedriver", options=options)

urlbase = 'https://www.imdb.com/title/'
urltail = '/reviews?ref_=tt_ql_3'


import pandas as pd

path_to_csv = "series.csv"

series = pd.read_csv(path_to_csv)

list_ids = series["imdbId"]

user_ratings = {}

for id_ in list_ids:
    url = urlbase + id_ + urltail
    driver.get(url)
    print("loading url {}...".format(url))
    #trouver le bouton responsable du chargement des résultats suivants
    loadmore = driver.find_element_by_id('load-more-trigger')
    while loadmore:
        #le cliquer
        loadmore.click()
        try:
            #attendre que ce bouton soit à nouveau visible (scroll) sur la page
            loadmore  = WebDriverWait(driver, 3).until(EC.visibility_of_element_located((By.ID, 'load-more-trigger')))
        except TimeoutException:
            break
    
    #quand on arrive ici on a chargé tous les avis possibles
    page = driver.page_source #récupérer la source de la page
    
    soup = BeautifulSoup(page, features="lxml")
    
    ratings = soup.find_all("span", {"class": "point-scale"})
    ratings_list = []
    users_list = []
        
    for elt in ratings:
        ratings_list.append(elt.previous_sibling.string)
        users_list.append(elt.parent.parent.next_sibling.next_sibling.next_sibling.next_sibling.span.a.string)
        
    for i in range(len(ratings_list)):
        user_ratings.setdefault(users_list[i], []).append((id_, ratings_list[i]))
            
    print("{} ratings found.".format(len(ratings_list)))
    
    #ajouter les ratings dans un csv
    
    #TODO