// -----------------------------------------------------------------------
//
//                                  CppCallingPython.cpp V 0.01
//
//                                (c) Brian Lynch February, 2015
//
// -----------------------------------------------------------------------

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

const int ArraySizeCHECK = 2000;

int load_data_file(std::string &infilename, std::vector<double> &id,
                                            std::vector<double> &ti,
                                            std::vector<double> &xx,
                                            std::vector<double> &yy,
                                            std::vector<double> &vx,
                                            std::vector<double> &vy,
                                            double **Array0,
                                            double **Array1,
                                            double **Array2,
                                            double **Array3){
   
   //Input string information
   std::string input_filename_s(infilename.c_str());
   std::ifstream input_file(input_filename_s.c_str(), std::ifstream::in);
    
    //Attempt to read the input file
   if(input_file.is_open()){
     
      std::cout << "Reading inputdata... FROM : " << infilename.c_str() << std::endl;
      double col1 = 0.0,
             col2 = 0.0,
             col3 = 0.0,
             col4 = 0.0,
             col5 = 0.0,
             col6 = 0.0;
             
      id.clear();
      ti.clear();
      xx.clear();
      yy.clear();
      vx.clear();
      vy.clear();
     
      while(input_file.good()){
        
         input_file >> col1 >> col2 >> col3 >> col4 >> col5 >> col6;
        
         if(!input_file.eof()){ //Last line is not stored twice
           
            id.push_back(col1);
            ti.push_back(col2);
            xx.push_back(col3);
            yy.push_back(col4);
            vx.push_back(col5);
            vy.push_back(col6); 
         }
       
      }
     
   }else{
     
      std::cerr << "Error opening file:" << input_filename_s.c_str();
      std::cerr << std::endl;
      return (-1);
      
   }//Done attempting to read file
   
   input_file.close();
   
   for(int i = 0; i < ArraySizeCHECK; i++){
      
      if(i < id.size()){
         (*Array0)[i] = xx.at(i);
         (*Array1)[i] = yy.at(i);
         (*Array2)[i] = vx.at(i);
         (*Array3)[i] = vy.at(i);
      }
      
   }
   
   std::cout << "Done reading input data" << std::endl;
   
return 1;
}

int main (int argc, char *argv[]){
  
    PyObject *pModName  = NULL,
             *pModule   = NULL,
	     *pDict     = NULL,
	     *pFunc     = NULL,
	     *pArgs     = NULL,
	     *pValue    = NULL,
	     *pX        = NULL,
	     *pY        = NULL,
	     *pVx       = NULL,
	     *pVy       = NULL,
	     *pArgTuple = NULL;
	     
    //# of entries in each 1D array
    npy_intp dims[1] = {  };
    
    PyArrayObject *py_array = NULL;
    const char *pFuncName = NULL;
    
    // Parse the command line
    if((argc < 3) || (argc > 3)) {
        fprintf(stderr,
		"Usage: ./a.out pythfile(without .py ext) pythfuncname(no arguments)\n");
        fprintf(stderr,
                "Usage: ./a.out <pythfile> <pythfuncname>\n");
        return 1;
    }
    
   
   
   //Set the new python array dimension
   std::vector<double> id, ti, xx, yy, vx, vy;
   dims[0] = ArraySizeCHECK;
   const int ArraySize = vy.size();
   double *Array0 = new double[ArraySizeCHECK],
          *Array1 = new double[ArraySizeCHECK],
          *Array2 = new double[ArraySizeCHECK],
          *Array3 = new double[ArraySizeCHECK];
          
   

   // Initialize Python
   printf("Initializing Python\n");
   Py_SetProgramName(argv[0]);
   Py_Initialize ();
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
      exit(-1);
   }
    
   // Load the module object
   printf("Importing Module\n");
   pModule = PyImport_Import(pModName);
   if(NULL == pModule){
      printf("Error with pModule PyImport_Import %p\n",pModule);
      exit(-1);
   }
    
   // Read the function name string and send it to Python
   printf("Converting function string: %s\n",argv[2]);
   pFuncName = argv[2];
   if(NULL == pFuncName){
      printf("Error with pFuncName PyString_FromString %p\n",pFuncName);
      exit(-1);
   }
    
   pFunc = PyObject_GetAttrString(pModule, pFuncName);
   if(NULL == pFunc){
      printf("Error with pFunc PyObject_GetAttrString %p\n",pFunc);
      exit(-1);
   }

   // Initialize Numpy
   // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
   printf("Initializing Numpy library\n");
   PyInit_Numpy();
   
   //NOW READ THE DATA FROM FILE
   std::string Lin_filename("output/pstate_list.dat");
   std::string fileline, 
               Iin_filename;
   std::ifstream Lin_file;
   Lin_file.open(Lin_filename.c_str());
   int itr = 0;
   
   while(std::getline(Lin_file,fileline)){

      Iin_filename = fileline;
      load_data_file(Iin_filename,id,ti,xx,yy,vx,vy,&Array0,&Array1,&Array2,&Array3);
      
   //   for(int i=0; i < ArraySizeCHECK; i++){
   //      std::cout << Array1[i] << " ";  
   //   }
   
      printf("Sending C++ arrays to Python\n");
      //Remeber to NOT free Array0-4 while Px, etc are still in existence.
      pX = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Array0);
      if(NULL == pX){
         printf("Error with pX PyArray_SimpleNewFromData %p\n",pX);
         exit(-1);
      }
      pY = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Array1);
      if(NULL == pY){
         printf("Error with pY PyArray_SimpleNewFromData %p\n",pY);
         exit(-1);
      }
      pVx = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Array2);
      if(NULL == pVx){
         printf("Error with pVx PyArray_SimpleNewFromData %p\n",pVx);
         exit(-1);
      }
      pVy = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, Array3);
      if(NULL == pVy){
         printf("Error with pVy PyArray_SimpleNewFromData %p\n",pVy);
         exit(-1);
      }   

      //Implement the module's namespace. This function never fails.
      pDict = PyModule_GetDict(pModule);

      //Returns uninitialize tuple of given length
      pArgTuple = PyTuple_New(4); //4 total arrays going in
   
      //Insert a reference to each pX object at appropriate position
      //within the pArgTuple. The reference to pX is STOLEN.
      PyTuple_SetItem(pArgTuple, 0, pX);
      PyTuple_SetItem(pArgTuple, 1, pY);
      PyTuple_SetItem(pArgTuple, 2, pVx);
      PyTuple_SetItem(pArgTuple, 3, pVy);

      //pFunc is a borrowed reference
      pFunc = PyDict_GetItemString(pDict, pFuncName); // pFunc is also a borrowed reference
 
      if(PyCallable_Check(pFunc)){
      
         pValue = PyObject_CallObject(pFunc, pArgTuple);
      
      }else{
      
         printf ("Function pFunc not callable !\n");
      
      }
      
      itr++;
      std::cout << "ITR #: " << itr << std::endl;
      
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
   Py_Finalize ();                                   
    
   //Now the memory can be freed after Py_XDECREF has been called
   delete[] Array0;
   delete[] Array1;
   delete[] Array2;
   delete[] Array3;

return 1;
}
