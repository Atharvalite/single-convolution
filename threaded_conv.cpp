#include <iostream>
#include<bits/stdc++.h>
#include<thread>
#include "threaded_conv.h"

using namespace std;



void compute_dot_product(vector<int> flat_kernel, vector<int> img_block, int **output, int offx, int offy, int size){
    int s=0;
    for(int i=0;i<size;i++){
        s+=flat_kernel[i]*img_block[i];
    }
    output[offx][offy] = s;
}

int** threaded_convolution(vector<vector<int>> matrix, vector<vector<int>> kernel){
    int inp_shape = matrix.size();
    int ker_shape = kernel.size();
    
    if(ker_shape>inp_shape){
        throw runtime_error("The size of kernel is larger than the size of matrix");
    }

    int op_shape = (inp_shape - ker_shape) + 1;

    // int *flatten_kernel = new int[ker_shape*ker_shape];
    vector<int> flatten_kernel(ker_shape*ker_shape);

    for(int i=0;i<ker_shape;i++){
        for(int j=0;j<ker_shape;j++){
            flatten_kernel[i*ker_shape+j] = kernel[i][j];
        }
    }

    // initialize output matrix
    int **output = new int*[op_shape];
    for(int i=0;i<op_shape;i++){
        output[i] = new int[op_shape];
    }
    //initializing im2col (op_shape*op_shape)x(ker_shape*ker_shape)
    int imcolx = op_shape*op_shape;
    int imcoly = ker_shape*ker_shape;

    // int **im2col = new int*[imcolx];
    // for(int i=0;i<imcolx;i++){
    //     im2col[i] = new int[imcoly];
    // }

    vector<vector<int>> im2col(imcolx, vector<int>(imcoly, 0));

    vector<std::thread*> threads;
    int NUM_Threads = 8;

    // threads.reserve(NUM_Threads);
    for(int i=0;i<op_shape;i++){
        for(int j=0;j<op_shape;j++){
            //forming flattened window
            for(int l=0;l<ker_shape;l++){
                for(int m=0;m<ker_shape;m++){
                    im2col[op_shape*i+j][l*ker_shape+m] = matrix[i+l][j+m];
                }
            }
            // compute_dot_product(flatten_kernel, im2col[op_shape*i+j], output, i, j, imcoly);
            // threads.push_back(std::move(td));
            if(threads.size()>=8){
                for (int i = 0; i < threads.size(); i++) {
                    threads[i]->join();
                }
                threads.clear();
            }
            
            std::thread* td = new std::thread(compute_dot_product, flatten_kernel, im2col[op_shape*i+j], output, i, j, imcoly);

            threads.push_back(std::move(td));

        }
    }

    return output;
}