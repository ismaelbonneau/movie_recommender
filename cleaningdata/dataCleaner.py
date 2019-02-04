#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:30:12 2019

@author: Ismael Bonneau & Issam Benamara
"""
import parse
import rename

def cleanData():
    """ External dataset Path """
    
    with open('dataPath.txt') as f:
        lines = f.readlines()
        path = lines[0].strip('\n')
    
    
    """ parsing/cleaning files """
    # extract only meaningful ( linguistic ) text from all the data
    # removing empty directories
    # remove non english episodes
    
    parse.cleanData(path)
    rename.rename(path)

