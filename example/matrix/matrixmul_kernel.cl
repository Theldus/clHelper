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

/* 
 * matrixmul_kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
/* OpenCL Kernel. */
__kernel void
matrixMul(__global double* c, 
          __global double* a, 
          __global double* b, 
          int width)
{
	double sum = 0;
	
	int row = get_global_id(1);
	int col = get_global_id(0);
   
	if (row < width && col < width)
	{
		/*
		 * Value stores the element that is computed by the thread.
		 */		
		for (int k = 0; k < width; k++)
			sum += a[row * width + k] * b[k * width + col];
	 
		/*
		 * Write the matrix to device memory each thread
		 * writes one element.
		 */
		c[row * width + col] = sum;
	}
}
