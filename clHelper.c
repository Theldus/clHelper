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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "clHelper.h"

#ifdef CL_DEBUG
	int debug_on = 1;
#else
	int debug_on = 0;
#endif

/**
 * Rounds up to the next power of two.
 * @param target Target number to be rounded.
 * @returns The next power of two.
 */
static size_t roundPower(size_t target)
{
	target--;
	target |= target >> 1;
	target |= target >> 2;
	target |= target >> 4;
	target |= target >> 8;
	target |= target >> 16;
	target++;
	return (target);
}

/**
 * Reads the kernel from a specified file.
 * @param path File to be read.
 * @param buf Memory buffer.
 * @returns Returns a positive number if success and a negative
 * number otherwise.
 */
int clhLoadKernel(struct cl_helper_context *chc, char const *path,
	char const *kernel_name)
{
	FILE   *fp;
	size_t fsz;
	long   off_end;
	int    rc;
	char   *buf;
	int    err;
	
	/* Set buffer. */
	buf = chc->buffer;

	/* Open the file */
	fp = fopen(path, "r");
	if (fp == NULL)
		return (-CLH_FILE_ERROR);

	/* Seek to the end of the file */
	rc = fseek(fp, 0L, SEEK_END);
	if (rc != 0)
		return (-CLH_FILE_ERROR);

	/* Byte offset to the end of the file (size) */
	if ((off_end = ftell(fp)) < 0)
		return (-CLH_FILE_ERROR);

	fsz = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	buf = malloc(fsz+1);
	if (buf == NULL)
		return (-CLH_FILE_ERROR);

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Read file into buffer */
	if (fread(buf, 1, fsz, fp) != fsz)
	{
		free(buf);
		return (-CLH_FILE_ERROR);
	}

	/* Close the file */
	if (fclose(fp) == EOF)
	{
		free(buf);
		return (-CLH_FILE_ERROR);
	}

	/* Make sure the buffer is NUL-terminated, just in case */
	buf[fsz] = '\0';
	chc->buffer = buf;

	/**
	 * Now, we have to prepare the environment.
	 */
	chc->program = clCreateProgramWithSource(chc->context, 1,
		(const char **)&chc->buffer, NULL, &err);

	if (!chc->program)
	{
		fprintf(stderr, "clHelper: Failed to create compute program!\n");
		return (-CLH_NOT_COMP_PROG);
	}

	/* Build the program executable. */
	if ( (clBuildProgram(chc->program, 0, NULL, NULL, NULL, NULL)) != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		
		fprintf(stderr, "clHelper: Failed to build program executable!\n");
		clGetProgramBuildInfo(chc->program, chc->device_id, CL_PROGRAM_BUILD_LOG,
			sizeof(buffer), buffer, &len);
		
		fprintf(stderr, "%s\n", buffer);
		exit(1);
	}

	/* Create the compute kernel in the program we wish to run. */
	chc->kernel = clCreateKernel(chc->program, kernel_name, &err);
	if (!chc->kernel || err != CL_SUCCESS)
	{
		fprintf(stderr, "clHelper: Failed to create compute kernel!\n");
		exit(1);
	}
	
	return (CLH_OK);
}

int clhStartContext(struct cl_helper_context *chc)
{
	int err;
	cl_uint platformCount;
	cl_platform_id* platform_ids;
	
	/* Clean the context structure. */
	memset(chc, 0, sizeof(struct cl_helper_context));
	
#ifdef CL_DEBUG
	fprintf(stderr, "Initializing OpenCL device...\n"); 
#endif

	clGetPlatformIDs(0, 0, &platformCount);
	
#ifdef CL_DEBUG
	fprintf(stderr, "Found %d platforms(s)...\n\n", platformCount);
#endif
	
	/*
	 * Get the platforms available.
	 */
	platform_ids = malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platform_ids, NULL);
	
	for (int i = 0; i < platformCount; i++)
	{
		char *info;
		size_t infoSize;
	
		cl_uint deviceCount;
		cl_device_id* devices;

#ifdef CL_DEBUG		
		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL, &infoSize);
		info = malloc(infoSize);
		clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, infoSize, info, NULL);
		
		fprintf(stderr, "Platform #%d: %s\n", i, info);
		free(info);
#endif		
		
		/**
		 * For each platform, iterate over the devices.
		 */
		clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
        
        for (int j = 0; j < deviceCount; j++)
        {
        	char *device_name;
        	size_t device_size;

        	cl_device_type device_type;
			size_t max_group_size;
			cl_uint max_items_dimensions;
			size_t max_work_item_size[3];
			cl_uint global_work_size;
			
			/* Device name. */
#ifdef CL_DEBUG
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &device_size);
			device_name = malloc(device_size);
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, device_size, device_name, NULL);
			fprintf(stderr, "  Device #%d: %s\n", j, device_name);
			free(device_name);
#endif
			
			/* Device type */
			clGetDeviceInfo(devices[j], CL_DEVICE_TYPE,
				sizeof(device_type), &device_type, NULL);
#ifdef CL_DEBUG
			switch (device_type)
			{
				case CL_DEVICE_TYPE_CPU:
					fprintf(stderr, "    Device type: CL_DEVICE_TYPE_CPU\n");
					break;
					
				case CL_DEVICE_TYPE_GPU:
					fprintf(stderr, "    Device type: CL_DEVICE_TYPE_GPU\n");
					break;
					
				case CL_DEVICE_TYPE_ACCELERATOR:
					fprintf(stderr, "    Device type: CL_DEVICE_TYPE_ACCELERATOR\n");
					break;
				
				default:
					fprintf(stderr, "    Device type: NOT_RECOGNIZED\n");
					break;
			}
#endif
			
			/* Max number of work-items in a work-group. */
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(max_group_size), &max_group_size, NULL);
#ifdef CL_DEBUG
			fprintf(stderr, "    Max work-items: %zd\n", max_group_size);
#endif			
			/* Max work-item dimensions. */
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
				sizeof(max_items_dimensions), &max_items_dimensions, NULL);
#ifdef CL_DEBUG
			fprintf(stderr, "    Max work-items dimensions: %d\n", max_items_dimensions);
#endif			
			/* Max work-item size for dimensions. */
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES,
				sizeof(max_work_item_size), &max_work_item_size, NULL);
#ifdef CL_DEBUG
			fprintf(stderr, "    Max work-items size for dimensions: (%zd, %zd, %zd)\n",
				max_work_item_size[0], max_work_item_size[1], max_work_item_size[2]);
#endif			
			
			/* Global work size. */
			clGetDeviceInfo(devices[j], CL_DEVICE_ADDRESS_BITS,
				sizeof(global_work_size), &global_work_size, NULL);
#ifdef CL_DEBUG		
			fprintf(stderr, "    Global work size: %d bits\n", global_work_size);
#endif			
			
			/**
			 * Checks if the current device is a GPU, if so, set this as
			 * the device that will be used.
			 */
			if (device_type == CL_DEVICE_TYPE_GPU)
			{
				/* Some context. */
				chc->device_id = devices[j];
				
				/* Device info. */
				chc->device_type = device_type;
				chc->max_group_size = max_group_size;
				chc->max_items_dimensions = max_items_dimensions;
				chc->max_work_item_size[0] = max_work_item_size[0];
				chc->max_work_item_size[1] = max_work_item_size[1];
				chc->max_work_item_size[2] = max_work_item_size[2];
				chc->global_work_size = global_work_size;
				
				if (!debug_on)
				{
					free(devices);
					free(platform_ids);
					goto device_check;
				}
			}
		}
		free(devices);
	}
	free(platform_ids);
	
device_check:
	if (chc->device_id == NULL)
	{
		fprintf(stderr, "clHelper: No GPUs were found in the system\n");
		return (-CLH_GPU_NOT_FOUND);
	}
	
	/* Create a compute context. */
	chc->context = clCreateContext(0, 1, &chc->device_id, NULL, NULL, &err);
	if (!chc->context)
	{
		fprintf(stderr, "clHelper: Failed to create a compute context!\n");
		return (-CLH_NOT_COM_CONT);
	}
	
	/* Create a command queue. */
	chc->command_queue = clCreateCommandQueue(chc->context, chc->device_id,
		CL_QUEUE_PROFILING_ENABLE, &err);

	if (!chc->command_queue)
	{
		fprintf(stderr, "clHelper: Failed to create a command queue!\n");
		return (-CLH_NOT_COM_QUEUE);
	}
	
	return (CLH_OK);
}

/**
 * Sets the local work size, i.e: the block size.
 * @param x X-Axis.
 * @param y Y-Axis.
 * @param z Z-Axis.
 * @returns Returns a positive number if success and a negative
 * number otherwise. 
 */
int clhSetBlockSize(struct cl_helper_context *chc, size_t x, size_t y, size_t z)
{
	int dimensions;
	size_t xx = (x == 0) ? 1 : x;
	size_t yy = (y == 0) ? 1 : y;
	size_t zz = (z == 0) ? 1 : z;
	
	/* Check bounds. */
	if (xx*yy*zz > chc->max_group_size  || xx > chc->max_work_item_size[0] ||
		yy > chc->max_work_item_size[1] || zz > chc->max_work_item_size[2])
	{
		fprintf(stderr, "clHelper: Invalid work-item/block size!!\n");
		return (-CLH_INV_WORK_ITEM);
	}
	
	/* Check first dimension. */
	if (!x)
	{
		fprintf(stderr, "clHelper: Invalid dimension, X should be at least 1\n");
		return (-CLH_INV_DIM);
	}
	
	/* Get number of dimensions. */
	dimensions = 1;
	if (y != 0) dimensions++;
	if (z != 0) dimensions++;
	chc->dimensions = dimensions;

	/* Allocate and set local work sizes. */
	if (chc->localWorkSize != NULL)
		free(chc->localWorkSize);
	
	chc->localWorkSize = malloc(sizeof(size_t) * dimensions);
	chc->localWorkSize[0] = roundPower(x);
	if (y)
		chc->localWorkSize[1] = roundPower(y);
	if (z)
		chc->localWorkSize[2] = roundPower(z);
		
	return (CLH_OK);
}

/**
 * Sets the local work size. It's the same thing as the function
 * above, just an alias.
 * @param chc Context.
 * @param x X-Axis.
 * @param y Y-Axis.
 * @param z Z-Axis.
 * @returns Returns a positive number if success and a negative
 * number otherwise.
 */
int clhSetLocalSize(struct cl_helper_context *chc, size_t x, size_t y, size_t z)
{
	return clhSetBlockSize(chc, x, y, z);
}

/**
 * Configures the global work size, i.e: the grid size.
 * @param chc Context.
 * @param x X-Axis.
 * @param y Y-Axis.
 * @param z Z-Axis.
 * @returns Returns a positive number if success and a negative
 * number otherwise.
 */
int clhSetGridSize(struct cl_helper_context *chc, size_t x, size_t y, size_t z)
{
	/* Check if local size was already set. */
	if (chc->dimensions <= 0)
	{
		fprintf(stderr, "clHelper: The grid size should be set *after*"
			" blockSize!\n");
		return (-CLH_INV_GRID);
	}
	
	int dimensions;
	/* Check first dimension. */
	if (!x)
	{
		fprintf(stderr, "clHelper: Invalid dimension, X should be at least 1\n");
		return (-CLH_INV_DIM);
	}
	
	/* Get number of dimensions. */
	dimensions = 1;
	if (y != 0) dimensions++;
	if (z != 0) dimensions++;
	
	if (chc->dimensions != dimensions)
	{
		fprintf(stderr, "clHelper: Grid size should be the same dimension as"
			" the block size!\n");
		return (-CLH_INV_DIM);
	}
	
	/* Allocate and set local work sizes. */
	if (chc->globalWorkSize != NULL)
		free(chc->globalWorkSize);
	
	chc->globalWorkSize = malloc(sizeof(size_t) * dimensions);
	chc->globalWorkSize[0] = roundPower(chc->localWorkSize[0] * x);
	if (y)
		chc->globalWorkSize[1] = roundPower(chc->localWorkSize[1] * y);
	if (z)
		chc->globalWorkSize[2] = roundPower(chc->localWorkSize[2] * z);
		
#ifdef CL_DEBUG
	fprintf(stderr, "\nLocal size: ( ");
	for (int i = 0; i < dimensions; i++)
		fprintf(stderr, "%zd ", chc->localWorkSize[0]);
	fprintf(stderr, ")\n");
	
	fprintf(stderr, "Global size: ( ");
	for (int i = 0; i < dimensions; i++)
		fprintf(stderr, "%zd ", chc->globalWorkSize[0]);
	fprintf(stderr, ")\n");
#endif

	return (CLH_OK);
}

/**
 * Sets the global size, i.e: the total number of work-items in each axis.
 * @param chc Context.
 * @param x X-Axis.
 * @param y Y-Axis.
 * @param z Z-Axis.
 * @returns Returns a positive number if success and a negative
 * number otherwise.
 */
int clhSetGlobalSize(struct cl_helper_context *chc, size_t x, size_t y, size_t z)
{
	int dimensions;
	/* Check first dimension. */
	if (!x)
	{
		fprintf(stderr, "clHelper: Invalid dimension, X should be at least 1\n");
		return (-CLH_INV_DIM);
	}
	
	/* Get number of dimensions. */
	dimensions = 1;
	if (y != 0) dimensions++;
	if (z != 0) dimensions++;
	
	if (chc->dimensions != dimensions)
	{
		fprintf(stderr, "clHelper: Global size should be the same dimension as"
			" the local size!\n");
		return (-CLH_INV_DIM);
	}
	
	/* Allocate and set local work sizes. */
	if (chc->globalWorkSize != NULL)
		free(chc->globalWorkSize);
	
	chc->globalWorkSize = malloc(sizeof(size_t) * dimensions);
	chc->globalWorkSize[0] = roundPower(x);
	if (y)
		chc->globalWorkSize[1] = roundPower(y);
	if (z)
		chc->globalWorkSize[2] = roundPower(z);
		
	return (CLH_OK);
}

/**
 * Launch the kernel and measures the time spent.
 * @param chc Context.
 * @returns Returns a positive number if success and a negative
 * number otherwise. 
 */
int clhLaunchKernel(struct cl_helper_context *chc)
{
	int err;             /* Error code.        */
	cl_ulong time_start; /* Kernel start time. */ 
	cl_ulong time_end;   /* Kernel stop time.  */
	
	/* Launches the kernel. */
	err = clEnqueueNDRangeKernel(chc->command_queue, chc->kernel,
		chc->dimensions, NULL, chc->globalWorkSize, chc->localWorkSize,
		0, NULL, &chc->event);

	/* Wait finishes. */
	clWaitForEvents(1, &chc->event);
	clFinish(chc->command_queue);

	/* Execution time. */
	clGetEventProfilingInfo(chc->event, CL_PROFILING_COMMAND_START,
		sizeof(time_start), &time_start, NULL);

	clGetEventProfilingInfo(chc->event, CL_PROFILING_COMMAND_END,
		sizeof(time_end), &time_end, NULL);

	/* Save the time spent. */
	chc->time_ms = (time_end - time_start) / 1000000.0;

	if (err != CL_SUCCESS)
	{
		fprintf(stderr, "clHelper: Failed to execute kernel! %d\n", err);
		return (-CLH_KERN_FAIL);
	}
	
	return (CLH_OK);
}

/**
 * Release all the memory (or at least should be) spent in the context.
 * @param chc Context.
 * @returns Returns a positive number if success and a negative
 * number otherwise. 
 */
int clhReleaseContext(struct cl_helper_context *chc)
{
	/* OpenCL stuffs. */
	if (chc->program)
		clReleaseProgram(chc->program);
	if (chc->kernel)
		clReleaseKernel(chc->kernel);
	if (chc->command_queue)
		clReleaseCommandQueue(chc->command_queue);
	if (chc->context)
		clReleaseContext(chc->context);
		
	/* clHelper stuffs. */
	if (chc->globalWorkSize)
		free(chc->globalWorkSize);
	if (chc->localWorkSize)
		free(chc->localWorkSize);
	if (chc->buffer)
		free(chc->buffer);
		
	/* Clear the structure. */
	memset(chc, 0, sizeof(struct cl_helper_context));
	
	return (CLH_OK);
}
