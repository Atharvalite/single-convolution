# single-convolution

## Introduction
This GitHub repository provides an implementation of a single convolution operation on a 2-d matrix in 
1. Python Script
2. C++ application
3. Threaded C++ application

Convolution operation has its roots in Mathematics and Signal Processing. Convolution plays a fundamental role in various fields, such as image processing, deep learning, and audio analysis, enabling the extraction of valuable features from signals and data. Particularly 3 properties of convolution shine out in context of neural networks:
  1. Sparse Connections:  Convolutional layers in neural networks utilize sparse connections, where each output unit is connected to only a subset of input units
  3. Parameter Sharing: The same kernel matrix is applied throughout the image, allowing the network to learn spatially invariant and generalized features.
  4. Equivariate Nature: This property means that small changes in input reflects in the output, a very important property for object segmentation and detection tasks.


## Convolution Operation

A single convolution operation consists of an input matrix, and a kernel filter where:
```
size(kernel_filter) << size(input_matrix)
```
The kernel filter is slid over patches of input matrix in a sliding window fashion. At each index, the hardamord product (element-wise multiplication) of the patch and kernel_filter and the resulting products are summed to produce a single output value. This process is repeated for each position in the input, generating a new output matrix or feature map.

The shape of this new matrix is equal to ```(shape(input_matrix) - shape(kernel_filter))+1```.
The naive implementation of this operation would require 4 nested loops, 2 for traversing the new feature map dimensions, and 2 for traversing the patch and kernel_filter.

The inner loops can be parallelized by vectorizing it. Basically, the patches are first flattened into single dimensional array and concatenated with other patches to form a matrix im2col, the kernel_filter is also flattened, after that its a efficient matrix multiplication problem (k).Tx(im2col).
This parallelized algorithm is known as image-2-block(Im2Col).
It's essentially a tradeoff between speed and memory, as additional memory is required for im2col, but in practice the benefits outweigh the downsides.

### Python Script
An implementation of im2col is given in python script with the help of NumPy library. NumPy's linear algebra operations are very fast, the reason being it uses BLAS(basic linear algebra sub-routines, a very optimized low-level library) under the hood.

### C++
The naive implementation of C++ is written in tradiotional 4 nested loop form, vector<vector<int>> data type is used to store matrices.
The multithreaded C++ code is an attemp to parallelize vector dot products in im2col algorithm, with the help of thread library(first introduced in C++ v11). One of the foundational challenges was- at first the implementation would spawn a new thread for every dot product between two vectors, which in this case would lead to ```(output_matrix_size*output_matrix_size)``` number of threads, and as modern systems implement kernel-level threading, this means that so many context switches would eventually lead to performance downgrade, a thrashing effect. To mitigate this, number of threads are fixed,and would not exceed a certain number.
The cmake applicartion was unable to line the underlying classes related to thread library, one of them being POSIX pthread library. To mitigate this issue it is manually linked while build by appending to CMakeLists.txt the following: 
  ```
 target_link_libraries(Convolutions PRIVATE Threads::Threads)
  ```
 
## Run Using Cmake
  
  ```
 mkdir build
 cd build
 cmake ..
  ```
  
  This will create a "build" directory in your project folder and generate the necessary build files based on the CMakeLists.txt configuration.
  
  ```
  cmake --build .
  ```
  This command will invoke the appropriate build system (e.g., Make or Ninja) and compile your source files.
  ```
  ./Convolutions
  ```
  This will run a c++ application that reads input kernel matrix, and input matrix from a .txt file, perform convolution, measure time, and correctness of the function agains gold output for all three implementations.
