#pragma once

#include "ActionClassifier.h"
#include "Features.h"
#include <Eigen/Core>
#include <Eigen/Eigen>

using namespace Eigen;

typedef MatrixXd EigenVectors;
typedef std::pair<double, int> myPair;
typedef std::vector<myPair> PermutationIndices;	

extern MatrixXd imageToMatrix(Mat* image);
extern EigenVectors pca(MatrixXd xd);
extern void test_pca();
extern vector<vector<double>> generateData(int n);
extern MatrixXd vectorsToMat(vector<vector<double>>  vectors);