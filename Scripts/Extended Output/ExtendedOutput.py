# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:19:27 2026

@author: Austin
"""

import access2thematrix
import os
import numpy as np
import time
import png
import skimage
from natsort import natsorted
from PIL import Image
import multiprocessing
from multiprocessing import Process
from skimage.segmentation import chan_vese
global workerCount
global allAreaValues
from skimage.filters import threshold_otsu
import cv2
scan = []
allAreaValues = []
workerCount  = int(os.cpu_count())
#DIRECTORY MUST USE "/" AND NOT "\", include a "/" at the end of the file address if it is not already included
scanDirectory = "C:/Users/austi/Desktop/target/"
mtrx_data = access2thematrix.MtrxData()     
#this is what the workers will run
def LoopScan(l, sublists, lock):
    outputDirectoryName = f"Sequential_Subset_{l}"
    os.mkdir(outputDirectoryName)
    masks = []
    processScanList = []
    area = []
    for j in sublists[l]:
        processScanList.append(j)
    for k in processScanList:
        noScanTag = "_0001"
        if any(noScanTag in s for s in k):
            break
        dataIn = k
        traces, message = mtrx_data.open(dataIn)
        traces
        {0: 'forward/up', 1: 'backward/up'}
        im, message = mtrx_data.select_image('forward/up')
        scan = cv2.flip(im.data,0)
        def Subtract_Plane(scan):
            rows, cols = scan.shape
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = scan.flatten()
            A = np.column_stack((X_flat, Y_flat, np.ones_like(X_flat)))
            coeffs, _, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)
            fittedPlane = (coeffs[0] * X + coeffs[1] * Y + coeffs[2])
            leveledData = scan - fittedPlane 
            leveledData = leveledData - np.min(leveledData)
            leveledData = leveledData / np.max(leveledData)
            leveledOutput = leveledData
            leveledOutput = (255 * leveledData).astype(np.uint8)
            leveledOutputName = processScanList.index(f"{k}")
            png.from_array(leveledOutput, "L").save(f"{outputDirectoryName}/{leveledOutputName}L.png")
            return leveledData
        
        def Chan_Vese(processedData):
            otsuBinary = processedData > threshold_otsu(processedData)
            heightnm = 200
            widthnm = 200
            image = processedData
            area = []
            init = image > np.percentile(skimage.filters.gaussian(image, sigma = 10000), 92)
            cv = chan_vese(image,mu=0.01, lambda1=1, lambda2=0.9, tol=1e-9, max_num_iter= 100 , dt=1, init_level_set= init , extended_output=True,)
            width = image.shape[0]
            height = image.shape[1]
            mask = cv[0]
            numMasked = (width* height) - np.count_nonzero(mask)
            percentMasked = numMasked/(width*height)
            islandArea = (widthnm * heightnm) * percentMasked
            area.append(islandArea)
            area = np.matrix(area)
            masks.append(mask)
            initOut = Image.fromarray(init)
            maskOut = Image.fromarray(mask)
            otsuOut = Image.fromarray(otsuBinary)
            outputMaskName = processScanList.index(f"{k}")
            initOut.save(f"{outputDirectoryName}/{outputMaskName}I.png")
            maskOut.save(f"{outputDirectoryName}/{outputMaskName}M.png")
            otsuOut.save(f"{outputDirectoryName}/{outputMaskName}O.png")
            return area
        
        processedData = Subtract_Plane(scan)
        area = Chan_Vese(processedData)
        
        with lock:
            for i in area:
                outputArea = i
                print(str(outputArea)[2:-3])
                allAreaValues.append(str(outputArea)[2:-2])
            
    print("^^^ Output for sequential subset", l, "^^^\n")
    
if __name__ == "__main__":
    #gather a list of all scan file addresses
    lock = multiprocessing.Lock()
    def LoadScans(scanDirectory):
        global scanList
        global masterMTRX
        
        for root, dirs, files in os.walk(scanDirectory, topdown=True):
            files = natsorted(files)
            print(files)
            
            for filename in files:
                noScanTag = "_0001"
                #delete master matrix file for sublist prep
                scanListTemp = [scanDirectory + str(i) for i in files]
                scanList = [s for s in scanListTemp if noScanTag not in s]
                masterMTRX = list(set(scanListTemp) - set(scanList))
                #print("wMaster Matrix file:", masterMTRX)
                return scanList
            
    scanList  = LoadScans(scanDirectory)
    #print(len(scanList))
    
    #dynamically assign each process a set of scans
    def FileDistribution(workerCount, scanList):
        global sublistLength
        global sublists
        numSublists = int(workerCount)
        sublistLength = int(len(scanList) / numSublists)
        print("Number of sublists:", numSublists)
        print("Sublist length:", sublistLength)
        print("Excess Scans:", len(scanList) - (sublistLength* workerCount), "\n")
        sublists = [scanList[x:x+sublistLength] for x in range(0, len(scanList), sublistLength)]
        #re-inject master matrix address
        for entry in sublists:
            entry.append(masterMTRX)
        return scanList
    
    scanList  = FileDistribution(workerCount, scanList)
    processes = []
    processStart = time.time()
    for l in range(workerCount):
        time.sleep(5)
        worker = Process(target=LoopScan, args=(l, sublists,lock))
        worker.start()

    worker.join()
    processEnd = time.time()
    executionTime = processEnd - processStart
    global totalScans
    totalScans = workerCount * sublistLength
    avgTime = executionTime/totalScans
    time.sleep(10)
    print("Analysis Complete")
    print(totalScans, "scans analyzed in", np.round(executionTime, 2), "seconds")
    print("Effective processing time of ", np.round(avgTime, 2), "seconds per image\n")
        
        
