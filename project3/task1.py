# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:43:27 2020

@author: Mateusz.Wiza
"""

#Specify the input string
string = "knowledgeengineering"

## STEP 1 - Source reduction

#Get probabilities
used_str = []
prob = []

#Iterate through all symbols in string. If a symbol is in not used_str, then it 
#is seen first time, so the probability is 1/len(string). If symbol is in 
#used_str, it was seen before and the probability of this symbol is increased
#by 1/len(string)
for elem in string:
    if elem not in used_str:
        used_str.append(elem)
        prob.append([elem,1/len(string)])
    else:
        for i, x in enumerate(prob):
            if elem in x:
                prob[i][1] += 1/len(string)
                break
        
#Sort by probabilities
prob.sort(key = lambda tup: tup[1], reverse=True)

#Narrow down to 2 symbols
prob_narrow = prob[:]
history = []

#Take two lowest probailities and combine, repeat until only 2 are left
while len(prob_narrow)>2:
    new = [prob_narrow[-2][0]+prob_narrow[-1][0],(prob_narrow[-2][1]+prob_narrow[-1][1])]
    
    
    prob_narrow.pop(-1)
    prob_narrow.pop(-1)
    prob_narrow.append(new)
    
    history.append(new[0])
    
    prob_narrow.sort(key = lambda tup: tup[1], reverse=True)

#Keep track of the order in which probabilities were combined
history.append(prob_narrow[0][0]+prob_narrow[1][0])


## STEP 2 - Source coding
encodings = []
length = len(history)

#Start with the last element in history (all symbols) and compare it with 
#the previous element in history. The symbols are then divided into two groups: 
#one with symbols which do not appear in the previous and the other with symbols 
#which do appear. Both groups are then assigned either 0 or 1. 
for i in range(length):
    half1 = []
    half2 = []
    for elem in history[-1]:
        if len(history) > 1:
            if elem in history[-2]:
                half1.append(elem)
            else:
                half2.append(elem)
        else:
            half2.append(elem)
    
    if len(half1)>0 and len(half2)>0:
        encodings.append([half1,0])
        encodings.append([half2,1])
    elif len(history[-1]) == 2:
        encodings.append([half2[0],0])
        encodings.append([half2[1],1])
    elif len(history[-1]) == 1:
        encodings.append([half2[0],1])    
    elif len(half1)>0:
        encodings.append([half1,0])
    elif len(half2)>0:
        encodings.append([half2,1])
        
    history.pop(-1)  
 
#Create the final encoding of each symbol by combining 0s and 1s assigned 
#earlier to groups where this symbol appears.    
encodings_final = []

for i in range(len(prob)):
    code = ""
    letter = prob[i][0]
    
    for j in range(len(encodings)):
        if letter in encodings[j][0]:
            code += str(encodings[j][1])
    
    print(letter,code)
    encodings_final.append([letter, (prob[i][1]), code])