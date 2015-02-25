#pragma once

#include "ActionClassifier.h"
#include "Features.h"
#include <Eigen/Core>
#include <Eigen/Eigen>

using namespace Eigen;

typedef MatrixXd EigenVectors;
typedef std::pair<double, int> myPair;
typedef std::vector<myPair> PermutationIndices;	

extern EigenVectors pca(int newDim,MatrixXd xd);
extern MatrixXd getProjectionMatrix(int k,EigenVectors eigenVectors,PermutationIndices pi);
extern vector<double> applyProjection(vector<double> point, MatrixXd projection);
extern MatrixXd vectorsToMat(vector<vector<double>>  vectors);
extern void test_pca();
extern vector<vector<double>> generateData(int n);