CToPython
===========================
(c) Brian Lynch August, 2016
-------------------------------

This software was written for educational purposes. I wrote/hacked this 
code while teaching myself how to pass C arrays to Python for convenient
plotting. The AWESOME Cython project can be found at 

https://github.com/cython/cython

To Compile the code:
//g++ CppCallingPython.cpp -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7  -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O2 -Wall -lpython2.7

To-Do:
######

* Remove all hints of C++
* Add better command line parsing and improve the python PlotVField to
include the pair correlation function.