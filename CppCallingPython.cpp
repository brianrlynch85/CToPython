// -----------------------------------------------------------------------
//
//                                  CppCallingPython.cpp V 0.01
//
//                                (c) Brian Lynch February, 2015
//
// -----------------------------------------------------------------------


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>

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

const int ArraySize = 4;
double Array0 [] = {1.2, 3.4, 5.6, 7.8};
double Array1 [] = {1.2, 3.4, -5.6, -7.8};
double Array2 [] = {-1.2, -3.4, 5.6, 7.8};
double Array3 [] = {-1.2, -3.4, -5.6, -7.8};

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
	     
    npy_intp dims[1] = { 4 };
    
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

    // Initialize Python
    printf("Initializing Python\n");
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
    
    // Convert the module name to Python string
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

    printf("Sending C++ arrays to Python\n");
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

    pArgTuple = PyTuple_New(4); //4 total arrays going in
    /*
    for (i = 0; i < ArraySize; ++i) {
       pValue = PyFloat_FromDouble(Array0[i]);
       if (!pValue) {
          Py_DECREF(pXVec);
          Py_DECREF(pModule);
          fprintf(stderr, "Cannot convert array value\n");
          return 1;
       }
       PyTuple_SetItem(pX, i, pValue);
    }
    */
   // PyTuple_SetItem(pArgs, 0, py_array);
   PyTuple_SetItem(pArgTuple, 0, pX);
   PyTuple_SetItem(pArgTuple, 1, pY);
   PyTuple_SetItem(pArgTuple, 2, pVx);
   PyTuple_SetItem(pArgTuple, 3, pVy);

    pFunc = PyDict_GetItemString(pDict, pFuncName); // pFunc is also a borrowed reference

    if(PyCallable_Check(pFunc)) {
       pValue = PyObject_CallObject (pFunc, pArgTuple);
    }else{
       printf ("Function pFunc not callable !\n");
    }

    printf("Freeing the Python memory\n");
    Py_XDECREF(py_array);                               // Clean up
    Py_XDECREF(pModule);
    Py_XDECREF(pDict);
    Py_XDECREF(pFunc);
    Py_XDECREF(pModName);

    printf("Finalizing Python\n");
    Py_Finalize ();                                     // Finish the Python Interpreter

    return 0;
}