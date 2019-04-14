# clHelper   [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
A small two files 'library' that you help you with OpenCL annoying stuffs.

## Motivation
It's undeniable the power that OpenCL offers you, but honestly it's quite annoying to check the 
platforms, devices, reading the kernel (from an array, or text file), creating contexts and so on.

CUDA, in turn, has a very large focus on the kernel and with few instructions you already have something running on the GPU, the basic workflow would look something like this, right?
```
h_in = malloc(...)
h_out = malloc(...)

cudaMalloc(d_in...)
cudaMemcpy(d_in, h_in...)

cudaMalloc(d_out...)

dimGrid(...)
dimBlock(...)
kernel<<dimGrid, dimBlock>>(...)

cudaMemcpy(h_out, d_out...)
```
Why we don't have such thing in OpenCL?

## Library
clHelper tries to keep things simple and lets you focus on the kernel, that would be the most important thing
right? In order to do that, clHelper implements a couple of functions that:
- Finds and initializes the first GPU found
- Loads the kernel from a file
- Setup all the boring stuffs automagically: *clCreateContext, clCreateCommandQueue, clCreateProgramWithSource, clBuildProgram, clCreateKernel...*
- Launchs the kernel
- Automatically measures the time, ;-)

What clHelper don't do for you:
- Allocating memory
- Freeing memory
- Set kernel arguments

But this is easy enough, even in OpenCL.

## Getting started
You can get using OpenCL in a few steps:
1) First of all, you need to get a *context* and initialize it, not the OpenCL, but clHelper context:
```
struct cl_helper_context chc;
clhStartContext(&chc);
```
After that, you already have a GPU selected, a OpenCL context and a command queue working.

2) After the context, we need to load the kernel by specifying the kernel name and entry point:
```
clhLoadKernel(&chc, "kernel.cl", "kernelName");
```
3) Preparing the memory: clHelper does not provide any mechanism to allocate and/or copying memory from one place to another, just because it's simple enough, so you will have something like:
```
/* Allocate memory. */
cl_mem d_A;
d_in  = clCreateBuffer(chc.context, CL_MEM_READ_ONLY , size, NULL, NULL);
d_out = clCreateBuffer(chc.context, CL_MEM_READ_WRITE, size, NULL, NULL);

/* Copying memory. */
clEnqueueWriteBuffer(chc.command_queue, d_in, CL_TRUE, 0, size, h_in, 0, NULL, NULL);
```
Note in the above code that we use **chc.context** and **chc.command_queue**. The structure *cl_helper_context* just contains some data that OpenCL need to know. Since the allocation process is made by the cl_* functions (not clh_*), you need to pass the already initialized data to it. ;-).

4) Dimensions: Whether it's CUDA or OpenCL, you need to define the size of your problem. In CUDA, we have the concept of threads and blocks, in OpenCL we have the equivalent work-item and work-group respectively.

Since this library is primarily focused on those who wish to start programming in OpenCL (starting from CUDA), we have 4 functions that perform equivalent work:
```
/* Parameters with 0 value, indicates that the dimension specified does not exist. */
clhSetBlockSize(&chc, 32, 32, 0);
clhSetGridSize(&chc, 64, 64, 0);
```
and
```
clhSetLocalSize(&chc, 32, 32, 0);
clhSetGlobalSize(&chc, 2048, 2048, 0);
```
The first group is intended for those familiar with CUDA. The size of the GRID is used to calculate the size of the globalSize, so when using this group, make sure the blockSize has been set *before* the grid.

The second group is intended for those familiar with OpenCL, it is quite straightforward and requires no explanations.

5) Arguments: Every kernel needs arguments, and they are passed in the traditional way:
```
clSetKernelArg(chc.kernel, 0, sizeof(cl_mem), (void *)&d_in);
...
```
Pay attention to **chc.kernel**, used within the structure.

6) Launch the kernel: The kernel can be simply initialized with the following code snippet:
```
clhLaunchKernel(&chc);
```
The function will launch the kernel, wait for it finishes and measure the time elapsed, being stored as double in milliseconds in **chc.time_ms**.

7) Copying data back: Once your data has been processed, it's time to move the data from the GPU to the CPU again, which can be done with:
```
clEnqueueReadBuffer(chc.command_queue, d_out, CL_TRUE, 0, size, h_out, 0, NULL, NULL);
...
```
Again, note the **chc.command_queue**.

8) Releasing the memory: Having processed all that was needed, it is time to free up the memory used, which can be done in 3 steps: 1) free up host memory, 2) free memory from the device, 3) release the clHelper context, what it is shown below:
```
/* Step 1: Release host memory.          */
free(h_in);
free(h_out);

/* Step 2: Release device memory.        */
clReleaseMemObject(d_in);
clReleaseMemObject(d_out);

/* Step 3: Release the clHelper context. */
clhReleaseContext(&chc);
```
Done =), with 8 simple steps you can play with OpenCL, easy right?
A complete example can be found in the example/ folder.

## No magic
All data that clHelper needs to run are contained within the `struct cl_helper_context` structure. So if you want to invoke some OpenCL function directly, just use the data that is already there, at your disposal.

Of the data contained in `struct cl_helper_context`, the most noteworthy are:
```
size_t max_group_size;           /* Max work-group size, (equivalent to
                                    threads per block, in CUDA.  */

cl_uint max_items_dimensions;    /* Maximum dimention supported. */
size_t max_work_item_size[3];    /* Max work-items size per
                                    dimension.                   */

cl_uint global_work_size;        /* Global work size (equivalent to grid
	                                size, in CUDA.               */
```
These 4 data are relative to the current GPU and can offer you a hint of the card currently in use by the code.

As already mentioned, a simple but interesting field is:
```
/* Profilling. */
double time_ms;                  /* Time spent to execute the
                                    kernel.                     */
};
```
This field is always populated after running a kernel. It stores the runtime in milliseconds, so feel free to use it.

## Building
As you already have noticed, there are only 2 files: a clHelper.c and a clHelper.h, feel free to move them to the folder of your project and only include them in the building process. There's a Makefile in example/ that can be used as a suggestion to build.

----------------------------
That's it, if you liked, found a bug or wanna contribute, let me know, ;-).
