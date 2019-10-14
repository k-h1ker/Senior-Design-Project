import sys
import os
import csv
import glob
import pandas as pd
import numpy as np
import re #will help in string splitting at deisred delimtters
user_input = input("Enter the path of the directory of your files: ")
def addPathLength(user_input):
	#user_input = input("Enter the path of the directory of your files: ")
	#assert os.path.exists(user_input), "I did not find the file at, "+str(user_input) #check if file path exists
	for root, dirs, files in os.walk(user_input): #traverses all files in directory
		for file in files:
			if file.endswith('.txt'):
				filePath = os.path.join(root, file)
				#print(filePath)
				f = open(filePath,'r') 
				lines=f.readlines()
				f.close()
				lines.insert(0,'PathLength	') #in the very first row before everything add PathLength
				num_lines = 0
				with open(filePath, 'r') as f: #loop to open file and count number of lines
				    for line in f:
				        num_lines += 1
				folderPath=os.path.dirname(filePath) #gives name of current folder
				folderName=os.path.basename(folderPath)
				#print(folderName)
				length=0
				if folderName=="0.5mm":
					length="0.5	"
				elif folderName=="1mm":
					length="1	"
				elif folderName=="2mm":
					length="2	"
				elif folderName=="5mm":
					length="5	"
				for i in range(2,((num_lines*2)-1),2):#to go from one line to next, you increment by 2. 
					lines.insert(i,length) #inserts the coresponding length below the PathLength
				
				f = open(filePath, 'w')
				f.writelines(lines)
				f.close()
		
def delLines():
	assert os.path.exists(user_input), "I did not find the file at, "+str(user_input) #check if file path exists
	for root, dirs, files in os.walk(user_input): #traverses all files in directory
	        for file in files:
	                if file.endswith('.txt'):
	                        filepath = os.path.join(root, file)
	                        #print(filepath)
	                        f = open(filepath,'r') 
	                        lines = f.readlines() #Read the files lines
	                        f.close()
	                        del lines[3] #Delete bad rows
	                        del lines[2]
	                        del lines[0]
	                        f = open(filepath, 'w')
	                        f.writelines(lines) #rewrite the file without the removed lines
	                        f.close()
	addPathLength(user_input)
def addSample():
	assert os.path.exists(user_input), "I did not find the file at, "+str(user_input) #check if file path exists
	for root, dirs, files in os.walk(user_input): #traverses all files in directory
		for file in files:
			if file.endswith('.txt'):
				filePath = os.path.join(root, file)
				f = open(filePath,'r') 
				lines=f.readlines()
				f.close()
				lines.insert(0,'Sample	') #in the very first row before everything add PathLength
				num_lines = 0
				with open(filePath, 'r') as f: #loop to open file and count number of lines
				    for line in f:
				        num_lines += 1
				
				#print(file)
				nameFile=file.index(".") #finds the first occurence of dot usually will be at file extension
				file=file[0:nameFile] #take substring of file name from beginning to just before file extenstion
				file=file+"	"
				for i in range(2,((num_lines*2)-1),2):#to go from one line to next, you increment by 2. 
					lines.insert(i,file) #inserts the coresponding file name below the Sample 
				
				f = open(filePath, 'w')
				f.writelines(lines)
				f.close()
def combineAllFiles():
	assert os.path.exists(user_input), "I did not find the file at, "+str(user_input) #check if file path exists
	newCSV=open("Detector1.csv", 'a+') #creates newcsv file that can be used to append previousfiles
	writer = csv.writer(newCSV)
	hasHeader=True #this will be used in figuring out whether the csv file already has Sample PathLeength 1350 ..etc already at the first row
	if(os.stat("Detector1.csv").st_size == 0): 
		hasHeader=False
	else:
		hasHeader=True
	
	for root, dirs, files in os.walk(user_input): #traverses all files in directory
		for file in files:
			if file.endswith('.txt'):
				filePath = os.path.join(root, file)
				f = open(filePath,'r')
				lines=f.readlines() 
				f.close()
				words=[]
				for line in lines:
					words.append(line)
				#print("word: ",words)
				#print("lines: ",lines)
				#print("Length: ",len(lines))
				if(hasHeader==True): #if they have same header
					for i in range(1,len(words)): 	
						wordList=re.split(', |\t|\n',words[i]) #splits up string based on delimmiters
						wordList = list(filter(None, wordList)) #removes those cells in excel that have nothing in them especially at end of rowe there was empty info
						#print("wordList: ",wordList)
						if(len(wordList)!=0 or wordList!=None): #trying to remove the rows that have nothing in them but didnt work
							writer.writerow(wordList)
				else:
					for word in words: 	
						wordList=re.split(', |\t|\n',word)#splits up string based on delimmiters
						wordList = list(filter(None, wordList)) #removes those cells in excel that have nothing in them
						#print("wordList: ",wordList)
						if(word.strip()=='' or word.strip()=='\n'): #trying to remove the rows that have nothing in them but didnt work
							continue
						else:
							writer.writerow(wordList)
						hasHeader=True #once header has been inserted no need to repeat it
	newCSV.close()
	clearWhiteLines("Detector1.csv") #call the method to remove extra rows
#delLines()		
#addSample()
#combineAllFiles()
