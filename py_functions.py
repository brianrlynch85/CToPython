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

def PlotVField(x,y,vx,vy):
   #print '!! BEGIN FUNCTION PLOTVFIELD' 
  
   import matplotlib.pyplot as plt
   import matplotlib.gridspec as gridspec
   from numpy import histogram
   import PairCorrCalc as PCC
   
   if not plt.fignum_exists(1):
       plt.figure(figsize=(16,8))
       plt.ion()

   plt.clf()
   fig = plt.gcf()

   gs = gridspec.GridSpec(8,24)
  
      #******************************************** vector field                  
   #store current axis values for changing later                            
   axV = plt.subplot(gs[0:8,0:15])
   axV.grid(True) 

   Q = axV.quiver(x,y,vx,vy,color='blue')
   #Q.set_UVC(vx,vy)

   q = axV.quiverkey(Q, 0.5, 0.99, 1,'Velocity Vector Field',
                            coordinates = 'axes',
                            fontproperties={'weight': 'bold'})

   axV.axis((0,1024,0,768))
   axV.set_xlabel('x [Pixels]')
   axV.xaxis.labelpad = 0
   axV.set_ylabel('y [Pixels]')
   axV.yaxis.labelpad = 0

   #invert the y-axis for image coordinates
   axV.invert_yaxis() 

   #******************************************** velocity histogram
   # Vx histogram
   axVX = plt.subplot(gs[0:2,17:24])
   axVX.grid(True)

   histVX,binsVX = histogram(vx,bins=10)
   binWidthVX     = 0.7 * (binsVX[1] - binsVX[0])
   centerVX       = (binsVX[:-1] + binsVX[1:]) / 2
   pointsVX = axVX.plot(centerVX,histVX,'-r^',markersize=3.0)[0]

   axVX.set_xlabel('$V_x [pix/s]$')
   axVX.xaxis.labelpad = 0
   axVX.set_ylabel('Arb.Counts')
   axVX.yaxis.labelpad = 0

   # Vy histogram
   axVY = plt.subplot(gs[3:5,17:24])
   axVY.grid(True)

   histVY,binsVY = histogram(vy,bins=10)
   binWidthVY     = 0.7 * (binsVY[1] - binsVY[0])
   centerVY       = (binsVY[:-1] + binsVY[1:]) / 2
   pointsVY = axVY.plot(centerVY,histVY,'-r^',markersize=3.0)[0]

   axVY.set_xlabel('$V_y [pix/s]$')
   axVY.xaxis.labelpad = 0
   axVY.set_ylabel('Arb.Counts')
   axVY.yaxis.labelpad = 0

   #******************************************** pair correlation
   # g(r) histogram
   axPC = plt.subplot(gs[6:8,17:24])
   axPC.grid(True)

   NumPars = len(y)
   mbinsPC,histPC = PCC.PairCorr(x,y,75,1024,768,NumPars)
   pointsPC = axPC.plot(mbinsPC[1:],histPC,markersize=1.0)[0]

   axPC.set_xlabel('r [pixels]')
   axPC.xaxis.labelpad = 0
   axPC.set_ylabel('g(r)')
   axPC.yaxis.labelpad = 0
   axPC.grid()

   #******************************************** finish up
    
   plt.draw()
   plt.pause(0.5)

   #print '!! END FUNCTION PLOTVFIELD' 
   return

def printhello(x,y,vx,vy):

   import sys

   print '!!!!!!!!!!!!Hello From Python!!!!!!!!!!!!'
   sys.stdout.flush()
   return
