/** **********************************************************************
 *
 * @file
 * 
 * @author    (c) Brian Lynch <brianrlynch85@gmail.com>
 * @date      August, 2016
 * @version   0.01
 *
 * @brief Test implementation of sending data from C code to Python
 *
 * @todo Py_Finalize() causes a strange corrupted doubly-linked list????
 *
 * This file is subject to the terms and conditions defined in the
 * file 'License.txt', which is part of this source code package.
 *
 ************************************************************************/

//g++ CppCallingPython.cpp -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7  -fno-strict-aliasing -Wdate-time -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O2 -Wall -lpython2.7

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>

#include "numpy/arrayobject.h"

#if PY_MAJOR_VERSION >= 3
int PyInit_Numpy(){
  
   import_array();
}
#else
void PyInit_Numpy(){
  
   import_array();
	
}
#endif

const int MAXSIZE = 10000;

unsigned int load_data_file(const char *filename, double **X, double **Y,
                                               double **Vx, double **Vy){
   
   double temp_id = 0.0, temp_ti = 0.0,
          temp_x = 0.0 , temp_y = 0.0 ,
          temp_vx = 0.0, temp_vy = 0.0;

   unsigned int lines = 0;

   FILE *inputfile = NULL;

   if(inputfile = fopen(filename,"r")){

      while((fscanf(inputfile,"%lf %lf %lf %lf %lf %lf",&temp_id,&temp_ti,
                                                        &temp_x,&temp_y,
                                                        &temp_vx,&temp_vy)
                                                        != EOF)       &&
            (lines < MAXSIZE)){

         (*X)[lines] = temp_x;
         (*Y)[lines] = temp_y;
         (*Vx)[lines] = temp_vx;
         (*Vy)[lines] = temp_vy;
         ++lines;
      }

      fclose(inputfile);

   }else{

      printf("ERROR: Failed to open file: %s\n",filename);
      exit(EXIT_FAILURE);

   }
   
   printf("Done reading input data file containing %d lines.\n", lines);
   
return lines;
}

int main (int argc, char *argv[]){
  
    PyObject *pModName  = NULL, *pModule   = NULL,
	     *pDict     = NULL,
	     *pFunc     = NULL,
	     *pArgs     = NULL,
	     *pValue    = NULL,
	     *pX        = NULL, *pY        = NULL,
	     *pVx       = NULL, *pVy       = NULL,
	     *pArgTuple = NULL;
	     
    //# of entries in each 1D array
    npy_intp dims[1] = {  };
    
    PyArrayObject *py_array = NULL;
    const char *pFuncName = NULL;
    
    // Parse the command line
    if((argc < 3) || (argc > 3)) {
       printf("Usage: ./a.out pythfile(no .py ext) pythfuncname(no args)\n");
       printf("Usage: ./a.out <pythfile> <pythfuncname>\n");
       return 1;
    }
   
   //Set the new python array dimension
   double *X = (double *)malloc(MAXSIZE * sizeof(double)),
          *Y = (double *)malloc(MAXSIZE * sizeof(double)),
          *Vx = (double *)malloc(MAXSIZE * sizeof(double)),
          *Vy = (double *)malloc(MAXSIZE * sizeof(double));

   if (NULL == X || NULL == Y || NULL == Vx || NULL == Vy){
      printf("ERROR: Failed to initialize data arrays\n");
   } 

   // Initialize Python
   printf("Initializing Python\n");
   Py_SetProgramName(argv[0]);
   Py_Initialize();
   if(0 == Py_IsInitialized()){
      printf("ERROR: Python failed to initialize\n");
   }
    
   // Tell Python where to find the module we are going to use
   printf("Setting PYTHONPATH to current directory\n");
   PySys_SetArgv(argc, argv);
   PyObject *sys = PyImport_ImportModule("sys");
   PyObject *path = PyObject_GetAttrString(sys, "path");
   PyList_Append(path, PyString_FromString("."));
    
   // Convert the module name (from command line) to Python string
   printf("Converting module string: %s\n",argv[1]);
   pModName = PyString_FromString(argv[1]);
   if(NULL == pModName){
      printf("Error with pModName PyString_FromString %p\n",pModName);
      exit(EXIT_FAILURE);
   }
    
   // Load the module object
   printf("Importing Module\n");
   pModule = PyImport_Import(pModName);
   if(NULL == pModule){
      printf("Error with pModule PyImport_Import %p\n",pModule);
      exit(EXIT_FAILURE);
   }
    
   // Read the function name string and send it to Python
   printf("Converting function string: %s\n",argv[2]);
   pFuncName = argv[2];
   if(NULL == pFuncName){
      printf("Error with pFuncName PyString_FromString %p\n",pFuncName);
      exit(EXIT_FAILURE);
   }
    
   pFunc = PyObject_GetAttrString(pModule, pFuncName);
   if(NULL == pFunc){
      printf("Error with pFunc PyObject_GetAttrString %p\n",pFunc);
      exit(EXIT_FAILURE);
   }

   // Initialize Numpy
   // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
   printf("Initializing Numpy library\n");
   PyInit_Numpy();
   
   //NOW READ THE TEST DATA FROM FILE
   std::string Lin_filename("output/pstate_list.dat");
   std::string fileline, 
               Iin_filename;
   std::ifstream Lin_file;
   Lin_file.open(Lin_filename.c_str());
   int itr = 0;
   
   while(std::getline(Lin_file,fileline)){

      Iin_filename = fileline;
      dims[0] = load_data_file(Iin_filename.c_str(),&X,&Y,&Vx,&Vy);
   
      printf("Sending C++ arrays to Python\n");
      //Remeber to NOT free Array0-4 while Px, etc are still in existence.
      pX = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, X);
      if(NULL == pX){
         printf("Error with pX PyArray_SimpleNewFromData %p\n",pX);
         exit(EXIT_FAILURE);
      }
      pY = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Y);
      if(NULL == pY){
         printf("Error with pY PyArray_SimpleNewFromData %p\n",pY);
         exit(EXIT_FAILURE);
      }
      pVx = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Vx);
      if(NULL == pVx){
         printf("Error with pVx PyArray_SimpleNewFromData %p\n",pVx);
         exit(EXIT_FAILURE);
      }
      pVy = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Vy);
      if(NULL == pVy){
         printf("Error with pVy PyArray_SimpleNewFromData %p\n",pVy);
         exit(EXIT_FAILURE);
      }   

      // Implement the module's namespace. This function never fails.
      pDict = PyModule_GetDict(pModule);

      // Returns uninitialize tuple of given length
      pArgTuple = PyTuple_New(4); //4 total arrays going in
   
      // Insert a reference to each pX object at appropriate position
      // within the pArgTuple. The reference to pX is STOLEN.
      PyTuple_SetItem(pArgTuple, 0, pX);
      PyTuple_SetItem(pArgTuple, 1, pY);
      PyTuple_SetItem(pArgTuple, 2, pVx);
      PyTuple_SetItem(pArgTuple, 3, pVy);

      // pFunc is a borrowed reference
      pFunc = PyDict_GetItemString(pDict, pFuncName); // pFunc is also a borrowed reference
 
      if(PyCallable_Check(pFunc)){
      
         pValue = PyObject_CallObject(pFunc, pArgTuple);
      
      }else{
         printf("Function pFunc not callable !\n");
      }
      
      itr++;
      printf("Iteration #: %d\n",itr);
      
      Py_XDECREF(pX);
      Py_XDECREF(pY);
      Py_XDECREF(pVx);
      Py_XDECREF(pVy);
      
   }

   //Clean up the python reference counting
   printf("Freeing the Python memory\n");
   Py_XDECREF(py_array);                          
   Py_XDECREF(pModule);
   Py_XDECREF(pDict);
   Py_XDECREF(pFunc);
   Py_XDECREF(pModName);

   //Finalize the Python interpreter
   printf("Finalizing Python\n");
   Py_Finalize();                                   
    
   //Now the memory can be freed after Py_XDECREF has been called
   free(X);
   free(Y);
   free(Vx);
   free(Vy);

return(1);
}
