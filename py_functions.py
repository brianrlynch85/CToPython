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
   import pylab as py
   
   plt.clf()
   
   plt.ion()
   
   #load the quiver plot
   Q = py.quiver(col0,col1,col2,col3,color='blue')

   #name the quiver plot and add title
   q = py.quiverkey(Q, 0.5, 0.99, 1,'Velocity Vector Field',
                            coordinates = 'axes',
                            fontproperties={'weight': 'bold'})
                            
   #store current axis values for changing later                            
   fig1 = plt.gcf()
   ax1 = fig1.gca()
   py.axis((0,1024,0,768))
   ax1.set_xlabel('x [Pixels]')
   ax1.set_ylabel('y [Pixels]')

   #invert the y-axis for image coordinates
   ax1.invert_yaxis()
   plt.grid(True)  
   
   plt.draw()
   plt.pause(0.0001)

   #print '!! END FUNCTION PLOTVFIELD' 
   return

def printhello():

   print '!!!!!!!!!!!!Hello From Python Bitches!!!!!!!!!!!!'
   return
