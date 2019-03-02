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



driver.get('https://www.imdb.com/title/tt0056751/reviews?ref_=tt_ql_3')

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

from bs4 import BeautifulSoup

soup = BeautifulSoup(page, features="lxml")

ratings = soup.find_all("span", {"class": "point-scale"})
ratings_list = []
users_list = []
    
for elt in ratings:
    ratings_list.append(elt.previous_sibling.string)
    users_list.append(elt.parent.parent.next_sibling.next_sibling.next_sibling.next_sibling.span.a.string)
        
for i, u in enumerate(users_list):
    print(u, ratings_list[i])