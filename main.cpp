#include <iostream>
#include<bits/stdc++.h>
#include "naive_conv.h"
#include "threaded_conv.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;


vector<int> splitString(const string& line, char delimiter) {
    vector<int> values;
    stringstream ss(line);
    string token;

    while (getline(ss, token, delimiter)) {
        values.push_back(stoi(token));
    }

    return values;
}

// Function to read the file and store data in a matrix
vector<vector<int>> readFile(const string& filename) {
    ifstream file;
    file.open(filename);
    vector<vector<int>> matrix;
    string line;

    if (file.is_open()) {
        while (getline(file, line)) {
            vector<int> row = splitString(line, ' ');
            matrix.push_back(row);
        }

        file.close();
    } else {
        cout << "Unable to open file: " << filename << endl;
    }

    return matrix;
}


int main(int, char**) {
    vector<vector<int>> inp = readFile("/home/atharva/Desktop/Workspace/single_convolution/test_cases/input_matrix.txt");


    vector<vector<int>> kernel = readFile("/home/atharva/Desktop/Workspace/single_convolution/test_cases/kernel_matrix.txt");
    
    auto start_naive = std::chrono::high_resolution_clock::now();
    vector<vector<int>> ans = convolution(inp, kernel);
    auto end_naive = std::chrono::high_resolution_clock::now();
    
    chrono::duration<double> elapsed_naive = end_naive - start_naive;

    cout<<"Time Taken by Naive Convolution: "<<elapsed_naive.count()<<" seconds"<<endl;

    start_naive = std::chrono::high_resolution_clock::now();
    int** ans2 = threaded_convolution(inp, kernel);
    end_naive = std::chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed_thread = end_naive - start_naive;

    cout<<"Time Taken by Threaded Convolution: "<<elapsed_thread.count()<<" seconds"<<endl;


    string command = "/bin/python3 /home/atharva/Desktop/Workspace/single_convolution/naive_conv.py";
    system(command.c_str());
    return 0;
}
