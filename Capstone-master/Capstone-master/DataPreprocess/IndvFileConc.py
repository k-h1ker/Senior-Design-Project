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
def clearWhiteLines(fileName): #trying to remove white rows
	f = open(fileName, 'r')#open file
	lines=f.readlines()#read all lines
	f.close()
	word=[] #make a new list that can be edited
	for line in lines:
		if(line=='\n'): #if the line is just a new line continue
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
                
#delLines()		
#addSample()
#combineIndvFiles()
#ConvertIndvToCSV()
#deleteTxts()
#IndvEdits()
