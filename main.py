# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 02:12:52 2016

@author: PeDeNRiQue
"""

from mvpa2.suite import *

import random
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array);
    
def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed    
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data

def str_to_number(data):
    return[[float(j) for j in i] for i in data]

def convert(values):
    position = 0;
       
    for i in range(len(values)):
        if(values[i] > values[position]):
            position = i
    result = [0]*3
    result[position] = 1
    return result          
           
           
if __name__ == "__main__":
    
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    #data = normalize_data(file_array)
    data = {"input": file_array[:,:-1], "target":file_array[:,-1]}
    # store the names of the colors for visualization later on
    color_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
             
    som = SimpleSOMMapper((50,20), 50, learning_rate=0.05)
    
    som.train(data["input"])
    #pl.imshow(som.K, origin='lower')
    
    targets = data["target"]
    data = som(data["input"])
    
    #print(mapped)
    
    #data = np.transpose(data)    
    
    for ind in range(len(targets)):    
        if(targets[ind] == 0): 
            red, = plt.plot(data[ind][0], data[ind][1], 'ro',label="Íris Setosa")
        elif(targets[ind] == 1):
            blue, = plt.plot(data[ind][0], data[ind][1], 'bs',label="Íris Versicolor")
        else:
            green, = plt.plot(data[ind][0], data[ind][1], 'g^',label="Íris Virgínica")
    #first_legend = plt.legend(handles=[red,blue,green], loc=1)
    #ax = plt.gca().add_artist(first_legend)
    #plt.savefig("rna_pca.png", dpi = 100)
    plt.show()
    
    for ind in range(len(targets)):    
        if(targets[ind] == 0): 
            red, = plt.plot(data[ind][0], data[ind][1], 'ro',label="Íris Setosa")
        elif(targets[ind] == 1):
            blue, = plt.plot(data[ind][0], data[ind][1], 'ro',label="Íris Versicolor")
        else:
            green, = plt.plot(data[ind][0], data[ind][1], 'ro',label="Íris Virgínica")
    #first_legend = plt.legend(handles=[red,blue,green], loc=1)
    #ax = plt.gca().add_artist(first_legend)
    #plt.savefig("rna_pca.png", dpi = 100)
    plt.show()
               
    