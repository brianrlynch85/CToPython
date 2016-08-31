CToPython
===========================
(c) Brian Lynch August, 2016
-------------------------------

This software was written for educational purposes. I wrote/hacked this 
code while teaching myself how to pass C arrays to Python for convenient
plotting. The AWESOME Cython project can be found at 

https://github.com/cython/cython

To Compile the code:
   -g++ c_to_python.c *command line options given by the command "python2.7-config --cflags"* -lpython2.7
   -you may need to install matplotlib using sudo apt-get install python-matplotlib

To-Do:
######

* Remove all hints of C++
* Add better command line parsing and improve the python PlotVField to
include the pair correlation function.