/*
 * MIT License
 *
 * Copyright (c) 2018 Davidson Francis <davidsondfgl@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef CL_HELPER_H
#define CL_HELPER_H

#include <CL/cl.h>

/*
 * clHelper codes
 */
#define CLH_OK             0
#define CLH_GPU_NOT_FOUND  1
#define CLH_NOT_COM_CONT   2
#define CLH_NOT_COM_QUEUE  3
#define CLH_NOT_COMP_PROG  4
#define CLH_INV_WORK_ITEM  5
#define CLH_INV_DIM        6
#define CLH_INV_GRID       7
#define CLH_KERN_FAIL      8
#define CLH_FILE_ERROR     9

/**
 * Data stuff.
 */
struct cl_helper_context
{
	/* Kernel buffer. */
	char *buffer;

	/* Kernel contexts. */
	cl_device_id device_id;          /* Compute device id.      */
	cl_context context;              /* Compute context.        */
	cl_command_queue command_queue;  /* Compute command queue.  */
	cl_program program;              /* Compute program.        */
	cl_kernel kernel;                /* Compute kernel.         */
	cl_event event;                  /* Time.                   */
	
	/* Device data. */
   	cl_device_type device_type;      /* Device type.                 */
	size_t max_group_size;           /* Max work-group size, (equivalent to
	                                    threads per block, in CUDA.  */

	cl_uint max_items_dimensions;    /* Maximum dimention supported. */
	size_t max_work_item_size[3];    /* Max work-items size per
	                                    dimension.                   */

	cl_uint global_work_size;        /* Global work size (equivalent to grid
	                                    size, in CUDA.               */
	                                  
	/* Kernel data. */
	size_t *localWorkSize;           /* Local work array.           */
	size_t *globalWorkSize;          /* Global work array.          */
	int dimensions;
	
	/* Profilling. */
	double time_ms;                  /* Time spent to execute the
	                                    kernel.                     */
};

/* -- External declarations. -- */

/* Load the kernel given a source file. */
extern int clhLoadKernel(struct cl_helper_context *chc, char const *path,
	char const *kernel_name);

/* Starts the clHelper context. */
extern int clhStartContext(struct cl_helper_context *chc);

/* Sets the block size. */
extern int clhSetBlockSize(struct cl_helper_context *chc, size_t x, size_t y,
	size_t z);

/* Sets the local size. */
extern int clhSetLocalSize(struct cl_helper_context *chc, size_t x, size_t y,
	size_t z);

/* Sets the grid size. */
extern int clhSetGridSize(struct cl_helper_context *chc, size_t x, size_t y,
	size_t z);
	
/* Sets the global size. */
extern int clhSetGlobalSize(struct cl_helper_context *chc, size_t x, size_t y,
	size_t z);
	
/* Launches the kernel. */
extern int clhLaunchKernel(struct cl_helper_context *chc);

/* Releases the context. */
extern int clhReleaseContext(struct cl_helper_context *chc);

#endif /* CL_HELPER_H. */
