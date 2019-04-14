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

#include <stdio.h>
#include <clHelper.h>

int main()
{
	int err;
	int width, size;
	struct cl_helper_context chc;

	/* OpenCL device memory for matrices. */
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;
	
	width = 2048;
	size = width * width * sizeof(double);
	double* h_A = (double*) malloc(size);
	double* h_B = (double*) malloc(size);
	double* h_C = (double*) malloc(size);

	/* Initialize host memory. */
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < width; j++)
		{
			h_A[i * width + j] = i;
			h_B[i * width + j] = j;
		}
	}

	/* Start context. */
	clhStartContext(&chc);
	
	/* Load kernel from file. */
	clhLoadKernel(&chc, "matrixmul_kernel.cl", "matrixMul");
	
	/* Create the input and output arrays in device memory for our calculation. */
	d_C = clCreateBuffer(chc.context, CL_MEM_READ_WRITE, size, NULL, &err);
	d_A = clCreateBuffer(chc.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_A, &err);
	d_B = clCreateBuffer(chc.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_B, &err);
	
	/* Block and grid size. */
	clhSetBlockSize(&chc, 32, 32, 0);
	clhSetGridSize(&chc, 64, 64, 0);
	
	/* Kernel arguments. */
	clSetKernelArg(chc.kernel, 0, sizeof(cl_mem), (void *)&d_C);
	clSetKernelArg(chc.kernel, 1, sizeof(cl_mem), (void *)&d_A);
	clSetKernelArg(chc.kernel, 2, sizeof(cl_mem), (void *)&d_B);
	clSetKernelArg(chc.kernel, 3, sizeof(int), (void *)&width);
	
	/* Launch kernel. */
	clhLaunchKernel(&chc);
	
	/* Copy d_C to h_C. */
	clEnqueueReadBuffer(chc.command_queue, d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL);
	
	printf("Time spent: %.4f ms\n", chc.time_ms);

#if 0
	for(int i = 0; i < width; i++)
		for(int j = 0; j < width; j++)
			printf("\n c[%d][%d] = %f",i, j, h_C[i*width+j]);
#endif
	
	/* Release host memory. */
	free(h_A);
	free(h_B);
	free(h_C);
 
 	/* Release device memory. */
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_C);
	clReleaseMemObject(d_B);
	
	/* Release clHelper memory. */
	clhReleaseContext(&chc);
}
