'''** **********************************************************************
 *
 * @file
 * 
 * @author    (c) Brian Lynch <brianrlynch85@gmail.com>
 * @date      August, 2016
 * @version   0.01
 *
 * @brief Calculates the pair correlation function
 *
 * @todo Clean up
 *
 * This file is subject to the terms and conditions defined in the
 * file 'License.txt', which is part of this source code package.
 *
 ************************************************************************'''

#import numpy as np
from numpy import greater_equal,less_equal,where,empty,arange,sqrt,add,divide,subtract,histogram
import math
#import ParticleSorting as PS

def PairCorr(xpos,ypos,maxDist,imageWidth,imageHeight,Npar):
 
   # Find particles in the cropping area
   xcropL = greater_equal(xpos,maxDist)              # left boundary
   xcropR = less_equal(xpos,(imageWidth - maxDist))  # right boundary
   ycropT = greater_equal(ypos,maxDist)              # top boundary
   ycropB = less_equal(ypos,(imageHeight - maxDist)) # bottom boundary
   xyIndices = xcropL * xcropR * ycropT * ycropB        # cropped boundary

   # Find cropped indices
   xycropIndices = where(xyIndices)[0]
   # Npar  = NPar#len(p_indices)
   NcropPar = len(xycropIndices)
   print 'Total # of Particles:',Npar
   print '# of Particles within PairCorrCalc boundary:',NcropPar

   # Compute the distances between all particles.
   distancesM = empty([NcropPar,Npar])
   print 'Calculating inter-particle spacings...'

   for citr in xrange(0,NcropPar,1):
      i = xycropIndices[citr]
      xdist  = subtract(xpos,xpos[i])**2
      ydist  = subtract(ypos,ypos[i])**2
      rdist = sqrt(add(xdist,ydist))
      distancesM[citr,:] = rdist
   
   # Transform the distance matrix into a linear array
   distancesL = distancesM.flatten('C')

   print 'Filtering particles beyond maxDist...'
   distancesLF = distancesL[distancesL <= maxDist]

   imageCropArea = float(imageWidth - maxDist) * float(imageHeight - maxDist)
   numDens  = float(NcropPar) / imageCropArea
   print'Cropped # Density:',"{:2.1e}".format(numDens),'[# / Pix^2]'

   # Compute the histogram and set its parameters
   numBins = 2 * maxDist
   binWidth = float(maxDist)/(numBins)
   print 'Radial bin width:',binWidth,'[Pixels]'
   mBins = arange(binWidth,maxDist,binWidth)
   print '# of bins:',len(mBins)

   # Compute the histogram
   hist,mbinEdges = histogram(distancesLF,bins=mBins)

   # Normalize the histogram to make it g(r)
   hist = divide(hist,mBins[1:])
   hist = hist / (2.0 * math.pi * Npar * binWidth * numDens)

   return mBins,hist