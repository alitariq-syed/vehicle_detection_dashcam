# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:46:03 2022

@author: Ali
"""

import os
import glob
import numpy as np
    
import zipfile #import ZipFile
from os.path import basename


# path = "mmdetection-master/demo"
path = "mmdetection-master/runs/sliced"

# create a ZipFile object
# with ZipFile('sampleDir.zip', 'w') as zipObj:
#    # Iterate over all the files in directory
#    for folderName, subfolders, filenames in os.walk(path):
#        for filename in filenames:
#            #create complete filepath of file in directory
#            filePath = os.path.join(folderName, filename)
#            # Add file to zip
#            zipObj.write(filePath, basename(filePath))    

def count_files(path):
    count=0
    for rootdir, dirs, files in os.walk(path):
        for filename in files:
            count+=1
    return count



def zipfolder(foldername, target_dir, files_per_part,skip):   
    # print(skip)         
    # zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED) #for compression
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w') #store only
    rootlen = len(target_dir) + 1
    file_count=0
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            # print(file)
            if file_count<skip:
                file_count+=1
                # print(file_count)
                continue
            else:
                if file_count<skip+files_per_part:
                    fn = os.path.join(base, file)
                    zipobj.write(fn, fn[rootlen:])
                    file_count+=1
                    if file_count % 1000 == 0:
                        print("file_count: ", file_count)
                else:
                    continue
                

filecount = count_files(path)

no_parts = 5
files_per_part = np.ceil(filecount/no_parts)

print("files_per_part: ", files_per_part)
for part in range(no_parts):
    print("part: ", part)
    
    zipfolder(foldername='d:/demo_'+str(part), target_dir = path, files_per_part=files_per_part,skip=files_per_part*part)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    