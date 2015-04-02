import os

trainFile='train'
testFile='test'

def splitData(datasetDir,odd=1):
    dirNames=addPrefix(datasetDir)
    for dirName in dirNames:
	    fileNames=addPrefix(dirName)
	    for fileName in fileNames:
                subject=fileName.split("_")[1]
                subject=int(subject.replace("s",""))
                if(subject % 2 == odd):
                    os.remove(fileName)
                    print(fileName)
    return None
	

def addPrefix(dir_):
    return map(lambda s: dir_+"/"+s,os.listdir(dir_))
		
splitData(testFile,0)
