"""
##############################################################################
# Author: Brian Lynch
# Edited: 1/6/15
#
##############################################################################
"""

import numpy as np
  
def distanceMatrix(NumPars1, xA1, yA1, NumPars2, xA2, yA2):

   distancesM = np.empty([NumPars1,NumPars2])
  
   for i in xrange(0,NumPars1):
      xdist  = (xA1[i] - xA2)**2
      ydist  = (yA1[i] - yA2)**2
      distsq = np.add(xdist,ydist)
      distancesM[i,:] = np.sqrt(distsq)
  
   return distancesM

def sortMaxDist(distancesM, maxDist):
  
   maxMIndices = np.where(distancesM < maxDist)
  
   return maxMIndices
   
def sortMinDist(distancesM, minDist):
  
   minMIndices = np.where(distancesM >= minDist)
  
   return minMIndices
   
def sortNanDist(distancesM):

#   nanMIndices = ~np.isnan(distancesM)#np.where(distancesM != nana)
   nanMIndices = np.where(distancesM != distancesM)
     
   return nanMIndices
   
def matchParticles():

   return   
""" 
distancesM = np.empty([NumParsDemon,NumParsPTV])


   


#Filtering particle matching beyond max_dist...
distancesLF = distancesL[distancesL <= max_dist]
NumPars = len(distancesIndices[0])

lDemonS = lDemonI[np.divide(distancesIndices[0],NumParsPTV)]
lPTVS   = lPTVI[np.mod(distancesIndices[0],NumParsPTV)]

idDemonS = idDemon[np.divide(distancesIndices[0],NumParsPTV)]
idPTVS   = idPTV[np.mod(distancesIndices[0],NumParsPTV)]   
print 'Remaining (after spatial sorting) particle #: ', NumPars
 
##############################################################################
#Find the new sorted ID's as well as X and Y positions
 
xDemonS = np.mod(lDemonS,imageWidth)
yDemonS = np.divide(lDemonS,imageWidth)
xPTVS = np.mod(lPTVS,imageWidth)
yPTVS = np.divide(lPTVS,imageWidth)

"""