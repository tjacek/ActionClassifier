#pragma once

#include "ActionClassifier.h"
#include "Features.h"
#include <Eigen/Core>
#include <Eigen/Eigen>

using namespace Eigen;

typedef MatrixXd EigenVectors;

extern MatrixXd imageToMatrix(DepthImage image);
extern EigenVectors pca(MatrixXd xd);
extern void test_pca();
