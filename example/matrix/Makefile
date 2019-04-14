# MIT License
#
# Copyright (c) 2018 Davidson Francis <davidsondfgl@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

CC=gcc
CLHELPER_DIR   = $(CURDIR)/..
CLHELPER_SRC   = $(CLHELPER_DIR)/clHelper.c
CLHELPER_DEBUG = -DCL_DEBUG

# Operation system architecture
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include

ifeq ($(OS_SIZE),32)
	CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
	CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
endif

INCLUDE  =  -I $(CLHELPER_DIR)/ -I $(CUDA_INC_PATH)
CL_LIBS  =  OpenCL
CFLAGS   =  -Wall -Werror -O3 -Wno-unused-variable
CFLAGS  +=  $(INCLUDE) -std=c99 $(CLHELPER_DEBUG)
LIB      =  -l$(CL_LIBS) -L $(CUDA_LIB_PATH)

all: matrix

matrix:
	$(CC) $(CFLAGS) matrix.c $(CLHELPER_SRC) -o matrix $(LIB)

clean:
	rm -f matrix
