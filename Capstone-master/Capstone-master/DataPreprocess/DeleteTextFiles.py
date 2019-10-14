import sys
import os
import csv
import glob
import pandas as pd
import numpy as np
import re #will help in string splitting at deisred delimtters
def deleteTxts():
        directory = input("Folder with Txt's Folder:") #set input folder
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith('.txt'):
                        os.remove(f)
deleteTxts()
