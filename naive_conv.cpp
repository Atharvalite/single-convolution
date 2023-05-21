#include <iostream>
#include<bits/stdc++.h>
#include "naive_conv.h"

using namespace std;


vector<vector<int>> convolution(vector<vector<int>> matrix, vector<vector<int>> kernel){
    int inp_shape = matrix.size();
    int ker_shape = kernel.size();
    
    if(ker_shape>inp_shape){
        throw runtime_error("The size of kernel is larger than the size of matrix");
    }

    int op_shape = (inp_shape - ker_shape) + 1;

    vector<vector<int>>ouptput_mat(op_shape, vector<int>(op_shape));
    int s;
    for(int i=0;i<op_shape;i++){
        for(int j=0;j<op_shape;j++){
            s=0;
            for(int m=i;m<i+ker_shape;m++){
                for(int n=j;n<j+ker_shape;n++){
                    s+=matrix[m][n]*kernel[m-i][n-j];
                }
            }
            ouptput_mat[i][j] = s;
        }
    }
    return ouptput_mat;
    

}
