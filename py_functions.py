'''** **********************************************************************
 *
 * @file
 * 
 * @author    (c) Brian Lynch <brianrlynch85@gmail.com>
 * @date      August, 2016
 * @version   0.01
 *
 * @brief Test implementation functions in python (called from C)
 *
 * @todo Clean up
 *
 * This file is subject to the terms and conditions defined in the
 * file 'License.txt', which is part of this source code package.
 *
 ************************************************************************'''

def PlotVField(col0,col1,col2,col3):
  
   #print '!! BEGIN FUNCTION PLOTVFIELD' 
  
   import matplotlib.pyplot as plt
   import matplotlib.gridspec as gridspec
   import numpy as np
   import ParticleSorting as PS
   import PairCorrCalc as PCC
   
   plt.clf()
   
   plt.ion()
   
   fig = plt.gcf()
   gs = gridspec.GridSpec(8,24)
   #ax = plt.subplot(gs[0,0])
  
   #******************** vector field plot ********************#                   
   #store current axis values for changing later                            
   axV = plt.subplot(gs[0:5,0:15])
    #load the quiver plot
   Q = axV.quiver(col0,col1,col2,col3,color='blue')
   #Q.set_UVC(col2,col3)

   #name the quiver plot and add title
   q = axV.quiverkey(Q, 0.5, 0.99, 1,'Velocity Vector Field',
                            coordinates = 'axes',
                            fontproperties={'weight': 'bold'})

   axV.axis((0,1024,0,768))
   axV.set_xlabel('x [Pixels]')
   axV.set_ylabel('y [Pixels]')

   #invert the y-axis for image coordinates
   axV.invert_yaxis()
   axV.grid(True) 

   #******************** velocity histogram plot ********************#
	# Vx histogram
   axVX = plt.subplot(gs[0:2,17:24])
   axVX.grid(True)
   histVX,binsVX = np.histogram(col2,bins=10)
   binWidthVX     = 0.7 * (binsVX[1] - binsVX[0])
   centerVX       = (binsVX[:-1] + binsVX[1:]) / 2
   pointsVX = axVX.plot(centerVX,histVX,'-r^',markersize=3.0)[0]
   # Vy histogram
   axVY = plt.subplot(gs[3:5,17:24])
   axVY.grid(True)
   histVY,binsVY = np.histogram(col3,bins=10)
   binWidthVY     = 0.7 * (binsVY[1] - binsVY[0])
   centerVY       = (binsVY[:-1] + binsVY[1:]) / 2
   pointsVY = axVY.plot(centerVY,histVY,'-r^',markersize=3.0)[0]

   #******************** pair correlation plot ********************#
   # Compute the pair correlation function
	# Pair correlation function
   axPC = plt.subplot(gs[6:8,17:24])
   axPC.grid(True)
   #axPC.set_xlim([0,cropDist])
   #axPC.set_ylim([0.0,10.0])

   distM          = PS.distanceMatrix(len(col1),col1,col2,len(col1),col1,col2)                
   inMDistIndices = PS.sortMaxDist(distM,50)
   NumPars        = len(inMDistIndices[0])
   xS    = col1[inMDistIndices[1]]
   yS    = col2[inMDistIndices[1]]
   mbinsPC,histPC = PCC.PairCorr(col1,col2,50,1024,768,len(col1))
	# Now perform particle matching between Demon and PTV based on pairDist
	
	
   # Pair correlation function
   pointsPC = axPC.plot(mbinsPC[1:],histPC,markersize=1.0)[0]
   #PCline   = axPC.axvline(parImDim,0,5,linewidth=3,linestyle='dashed', color='m')
   axPC.set_xlabel('r [pixels]')
   axPC.set_ylabel('g(r)')
   axPC.grid()

   #********************Wrap things up********************#
    
   
   plt.draw()
   plt.pause(0.0001)

   #print '!! END FUNCTION PLOTVFIELD' 
   return

def printhello(col0,col1,col2,col3):

   import sys
   

   print '!!!!!!!!!!!!Hello From Python Bitches!!!!!!!!!!!!'
   sys.stdout.flush()
   return
