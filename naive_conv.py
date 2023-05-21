from threading import main_thread
import numpy as np
import time


def convolution(matrix: np.array, kernel: np.array) -> np.array:
    ker_shape = kernel.shape[0]
    inp_shape = matrix.shape[0]

    if ker_shape>inp_shape:
        raise Exception("The size of Kernel Matrix is larger than input matrix")
    
    op_shape = (inp_shape - ker_shape) + 1

    # Flattening the windows, to vectorize the convolution operation 
    r = []
    for i in range(0, op_shape):
        for j in range(0, op_shape):
            window = matrix[i:i+ker_shape, j:j+ker_shape]

            r.append(window.flatten())

    imcol = np.transpose(np.array(r)) 

    
    flat_kernel = kernel.flatten()

    flat_output = np.dot(flat_kernel, imcol)


    return flat_output.reshape((op_shape, op_shape))
    
matrix = np.loadtxt("/home/atharva/Desktop/Workspace/single_convolution/test_cases/input_matrix.txt")
kernel = np.loadtxt("/home/atharva/Desktop/Workspace/single_convolution/test_cases/kernel_matrix.txt")
gold_output = np.loadtxt("/home/atharva/Desktop/Workspace/single_convolution/test_cases/gold_output.txt")

start_time = time.time()

output = convolution(matrix=matrix, kernel=kernel)

end_time = time.time()

print(f"Execution time for Python code: {end_time-start_time} seconds")

flag = False
for i in range(gold_output.shape[0]):
    for j in range(gold_output.shape[1]):
        if output[i][j]!=gold_output[i][j]:
            flag = True
            print("Naive Convolution has failed the testcase")
            break
if not flag:
    print("Naive Convolution has passed the testcase")