#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests

url = "https://www.imdb.com/title/tt0056751/reviews?ref_=tt_ql_3"

requete = requests.get(url)
if requete.status_code == 200:
    page = requete.content
    
    soup = BeautifulSoup(page)
    
    ratings = soup.find_all("span", {"class": "point-scale"})
    ratings_list = []
    users_list = []
    
    for elt in ratings:
        ratings_list.append(elt.previous_sibling.string)
        users_list.append(elt.parent.parent.next_sibling.next_sibling.next_sibling.next_sibling.span.a.string)
        
    for i, u in enumerate(users_list):
        print(u, ratings_list[i])

else:
    print("url {} won't work. error {}".format(url, requete.status_code))