# -------------------------------------------------------------------------------------
#
#                               Makefile for CppToPython
#                                        V 0.01
#
#                            (c) Brian Lynch February, 2015
#
# -------------------------------------------------------------------------------------
# possible make options are:
#      "make clean"            to clean up all .o and temp files
#      "make mrclean"          to clean up all .o, temp files, and executables
#      "make all"              to compile everything

CC       := g++
DEBUG    := -g
CCFLAGS  := $(DEBUG) -Wall -std=c++0x
PYTFLAGS := -lpython2.7

IPYTFLAGS := -I/usr/include/python2.7/

DIR_BASE := $(CURDIR)

.PHONY: clean all dir_b myclean example_commands

all: dir_b
	@echo "   "
	@echo "SUCCESSFULlY COMPILED!"
	@echo "Copied executables into /bin"
	@echo "   "

clean:
	@rm -f $(DIR_BASE)/*.o
	@rm -f $(DIR_BASE)/*~
	@echo "   "
	@echo "Deleted *.o and *~ files"
	@echo "   "

mrclean: clean
	rm -f $(DIR_BASE)/CppCallingPython
	@echo "   "
	@echo "Deleted executable files"
	@echo "   "
	
example_commands:
	@echo "   "
	@echo "./CppCallingPython py_func PlotVField"
	@echo "   "

# --------------------------------------------------------------------------------------
# Directory base
# --------------------------------------------------------------------------------------

dir_b: $(DIR_BASE)/CppCallingPython

$(DIR_BASE)/CppCallingPython.o: $($@:.o=.cpp) $($@:.o=.h) 
	$(CC) -o -c $@ $< $(CCFLAGS)

$(DIR_BASE)/CppCallingPython:  $(DIR_BASE)/CppCallingPython.cpp        
	$(CC) -o $@ $^ $(CCFLAGS) $(IPYTFLAGS) -lpython2.7