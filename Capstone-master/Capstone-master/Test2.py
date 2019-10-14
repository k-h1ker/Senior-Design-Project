import sys
import os
import csv
import glob

import numpy as np
import pandas as pd

import re #will help in string splitting at deisred delimtters

user_input = input("Enter the path of the directory of your files: ")

#adds Path Length folder the files are located in
def addPathLength(user_input):
	#user_input = input("Enter the path of the directory of your files: ")
	#assert os.path.exists(user_input), "I did not find the file at, "+str(user_input) #check if file path exists
	for root, dirs, files in os.walk(user_input): #traverses all files in directory
		for file in files:
			if file.endswith('.txt'):
				filePath = os.path.join(root, file)
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

#deletes first couple rows that just have software information		
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

#add in what sample file it is
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
				
				nameFile=file.index(".") #finds the first occurence of dot usually will be at file extension
				file=file[0:nameFile] #take substring of file name from beginning to just before file extenstion
				file=file+"	"
				for i in range(2,((num_lines*2)-1),2):#to go from one line to next, you increment by 2. 
					lines.insert(i,file) #inserts the coresponding file name below the Sample 
				
				f = open(filePath, 'w')
				f.writelines(lines)
				f.close()

#puts all text files into one big CSV file can call it whatever you want
def combineAllFiles():
	assert os.path.exists(user_input), "I did not find the file at, "+str(user_input) #check if file path exists
	newCSV=open("Detector2.csv", 'a+') #creates newcsv file that can be used to append previousfiles can change Detector2.csv to something else
	writer = csv.writer(newCSV)
	hasHeader=True #this will be used in figuring out whether the csv file already has Sample PathLeength 1350 ..etc already at the first row
	if(os.stat("Detector2.csv").st_size == 0): #determine if there is a header (PathLength  FileName Wavelengths)
		hasHeader=False
	else:
		hasHeader=True
	for root, dirs, files in os.walk(user_input): #traverses all files in directory
		for file in files:
			if file.endswith('.txt'):
				filePath = os.path.join(root, file)
				f = open(filePath,'r')
				lines=f.readlines() #reads in all lines in file
				f.close()
				words=[]
				for line in lines:
					words.append(line) #append each line to array

				if(hasHeader==True): #if they have same header
					for i in range(1,len(words)): 	
						wordList=re.split(', |\t|\n',words[i]) #splits up string based on delimmiters
						wordList = list(filter(None, wordList)) #removes those cells in excel that have nothing in them especially at end of rowe there was empty info
						if(len(wordList)!=0 or wordList!=None): #trying to remove the rows that have nothing in them but didnt work
							writer.writerow(wordList)
				else:
					for word in words: 	
						wordList=re.split(', |\t|\n',word)#splits up string based on delimmiters
						wordList = list(filter(None, wordList)) #removes those cells in excel that have nothing in them
						if(word.strip()=='' or word.strip()=='\n'): #trying to remove the rows that have nothing in them but didnt work
							continue
						else:
							writer.writerow(wordList)
						hasHeader=True #once header has been inserted no need to repeat it
	newCSV.close()
	clearWhiteLines("Detector2.csv") #call the method to remove extra rows(every other row was empty)

def clearWhiteLines(fileName): #trying to remove white rows
	f = open(fileName, 'r')#open file
	lines=f.readlines()#read all lines
	f.close()
	word=[] #make a new list that can be edited
	for line in lines:
		if(line=='\n'): #if the line is just a new line continue else add that to the array so we have rows that are not empty in the file
			continue
		elif("NaN" in line):
			continue
		elif("-" in line):
			continue	
		else:
			word.append(line) #if the row has info add it to new List
	f=open(fileName,'w')
	f.writelines(word) #write in the new list into the file
	f.close()

def combineIndvFiles():
        file_paths = {}
        for root, dirs, files in os.walk(user_input):
            for f in files:
                if f.endswith('.txt'):
                    if f not in file_paths:
                        file_paths[f] = []
                    file_paths[f].append(root)


        for f, paths in file_paths.items():
            txt = []
            for p in paths:
                with open(os.path.join(p, f)) as f2:
                    txt.append(f2.read())
            with open(f, 'w') as f3:
                f3.write(''.join(txt))

def ConvertIndvToCSV():
        directory = input("INPUT Folder:") #set input folder
        output = input("OUTPUT Folder:") #set output folder

        txt_files = os.path.join(directory, '*.txt') #use input folder to set as directory

        for txt_file in glob.glob(txt_files):
            with open(txt_file, "r") as input_file: #open txt files
                in_txt = csv.reader(input_file, delimiter='\t') #read text files with a space indicating different values
                filename = os.path.splitext(os.path.basename(txt_file))[0] + '.csv' #make csv file with same name as txt file

                with open(os.path.join(output, filename), 'w') as output_file: #open file created from previous line
                    out_csv = csv.writer(output_file) #write into outpute file
                    out_csv.writerows(in_txt) #write the contents read from in_txt

def IndvEdits():
        directory = input("INPUT Folder:") #set input folder
        
        for root, dirs, files in os.walk(directory):
            for f in files:
                filename = f
                clearWhiteLines(filename)


def deleteTxts():
        directory = input("INPUT Folder:") #set input folder

        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith('.txt'):
                        os.remove(f)



#delLines()		
#addSample()
#combineAllFiles()
#combineIndvFiles()

#ConvertIndvToCSV
#flipRowCol()
#calVariance()
#ConvertIndvToCSV()
#deleteTxts()
#IndvEdits()
