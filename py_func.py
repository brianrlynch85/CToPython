# -----------------------------------------------------------------------
#
#                                      py_func.py V 0.01
#
#                                (c) Brian Lynch February, 2015
#
# -----------------------------------------------------------------------

def PlotVField(col0,col1,col2,col3):
  
   print '!! BEGIN FUNCTION PLOTVFIELD' 
   #print col0
   #print col1
   #print col2
   #print col3
  
   import matplotlib.pyplot as plt
   #import numpy as np
   #import time
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
   xmin,xmax,ymin,ymax = plt.axis()
   py.axis((0,1024,0,768))
   #gca().set_xlabel('x [Pixels]')
   #gca().set_ylabel('y [Pixels]')

   #invert the y-axis for image coordinates
  # gca().invert_yaxis()
  # grid(True)  
   
   plt.draw()
   plt.pause(0.0001)
   
   #show()

   #time.sleep(5.0)

   print '!! END FUNCTION PLOTVFIELD' 
   return
  
##############################################################################
def PlotXY(col0,col1,col2,col3):
  
   print '!! BEGIN FUNCTION PLOTXY' 
   #print col0
   #print col1
   #print col2
   #print col3

   import matplotlib.pyplot as plt
   import numpy as np

   
   plt.ion()
   pointsP, = plt.plot(0,1,'bo',markersize=4.0)
   pointsP.set_xdata(col0)
   pointsP.set_ydata(col1)
   plt.draw()
   plt.pause(0.1)
   
  # fig.canvas.draw()
 
   #plt.ion()
   #plt.clf()
   #plt.plot(col0,col1,'bo',markersize=4.0)
   #plt.pause(0.01)
   #plt.draw()

   print 'END PROGRAM PLOTXY'

   return

def printhello():
   print '!!!!!!!!!!!!Hello From Python Bitches!!!!!!!!!!!!'
   return
   
#a = [0,1,2,3]
#b = [-0,-1,-2,-3]
#PlotVField(a,b,a,b)
#b=[1,2,3,4]
#plt.plot(b,a)
