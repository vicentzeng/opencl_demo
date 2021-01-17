#define CL_TARGET_OPENCL_VERSION 120
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cassert>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

#define OPENCL_CHECK_ERRORS(ERR)        \
    if(ERR != CL_SUCCESS)                  \
    {                                      \
    cerr                                   \
    << "OpenCL error with code " << ERR    \
    << " happened in file " << __FILE__    \
    << " at line " << __LINE__             \
    << ". Exiting...\n";                   \
    exit(1);                               \
    }

#define MAX_DETECT_NUM (128)


// OpenCL kernel to perform an element-wise addition
const char *programSource = \
"__kernel \n \
void vecadd(__global int *A, \n \
__global int *B, \n \
__global int *C) \n \
{ \n \
// Get the work-item's unique ID \n \
int idx = get_global_id(0); \n \
\n \
// Add the corresponding locations of \n \
// 'A' and 'B', and store the reasult in 'C' \n \
C[idx] = A[idx] + B[idx]; \n \
} \n";

/**
 * vec_add : bufA + bufB = bufC
 */

int opencl_test_vec_add() {
    cout << "opencl_test E" << endl;
    // This code executes on the OpenCL host
    // Elements in each array
    const int elements = 512; // query my GPU's CL_DEVICE_MAX_WORK_GROUP_SIZE is 512
    // Compute the size of the data
    size_t datasize = sizeof(int) * elements;
    // Allocate space for input/output host data
    int *A = (int *)malloc(datasize); // Input array
    int *B = (int *)malloc(datasize); // Input array
    int *C = (int *)malloc(datasize); // Output array
    // Initialize the input data
    int i;
    for (i = 0; i < elements; i++){
        A[i] = i;
        B[i] = i;
        C[i] = 0;
    }
    // Use this to check the output of each API call
    cl_int status;
    // Get the first platforms
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    OPENCL_CHECK_ERRORS(status);
    // Get the first devices
    //cl_device_id device = new cl_device_id[3];
    cl_device_id* device_ids = new cl_device_id[MAX_DETECT_NUM];
    memset(device_ids, 0, MAX_DETECT_NUM * sizeof(cl_device_id));
    cl_uint real_devices_num = 0;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, MAX_DETECT_NUM, device_ids, &real_devices_num);
    if (real_devices_num != 0) {
        cout << "find cl devices cnt:" << real_devices_num << endl;
        cout << "device_ids[0]:" << device_ids[0] <<endl;
        cout << "device_ids[1]:" << device_ids[1] <<endl;
    }
    OPENCL_CHECK_ERRORS(status);
    // Create a context and associate it with the device
    cl_context context = clCreateContext(NULL, real_devices_num, device_ids, NULL, NULL, &status);
    OPENCL_CHECK_ERRORS(status);
    // Create a command-queue and associate it with device
    // cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, device_ids[0], NULL, &status);   //for opencl v2.0
    cl_command_queue_properties properties = 0; //CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    cl_command_queue cmdQueue = clCreateCommandQueue(context, device_ids[0], properties, &status);              //for opencl v1.2

    OPENCL_CHECK_ERRORS(status);
    // Allocate two input buffers and one output buffer for the three vectors in the vector addition
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
    OPENCL_CHECK_ERRORS(status);
    // Write data from the input arrays to the buffers
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_TRUE, 0, datasize, B, 0, NULL, NULL);
    OPENCL_CHECK_ERRORS(status);
    // Create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
    OPENCL_CHECK_ERRORS(status);
    // Build(compile) the program for the device
    status = clBuildProgram(program, real_devices_num, device_ids, NULL, NULL, NULL);
    OPENCL_CHECK_ERRORS(status);
    // Create the vector addition kernel
    cl_kernel kernel = clCreateKernel(program, "vecadd", &status);
    OPENCL_CHECK_ERRORS(status);
    // Set the kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    OPENCL_CHECK_ERRORS(status);
    // Define an incde space of work-items for execution
    // A work-group size is not required, but can be used.
    size_t indexSpaceSize[1], workGroupSize[1];
    // There are 'elements' work-items
    indexSpaceSize[0] = elements;
    workGroupSize[0] = elements;

    // Execute the kernel
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
    OPENCL_CHECK_ERRORS(status);

    // Read the device output buffer to the host output array
    status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);
    OPENCL_CHECK_ERRORS(status);
    // Free OpenCL resouces
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);
    // out the result
    cout << "A:" << endl;
    for (int i = 0; i < elements; i++) {
        cout << A[i] <<" ";
    }
    cout <<endl;
    cout << "B:" << endl;
    for (int i = 0; i < elements; i++) {
        cout << B[i] <<" ";
    }
    cout <<endl;
    cout << "C:" << endl;
    for (int i = 0; i < elements; i++) {
        cout << C[i] <<" ";
    }
    cout <<endl;
    // free host resouces
    free(A);
    free(B);
    free(C);
    delete [] device_ids;
    cout << "opencl_test X" << endl;
return 0;
}
