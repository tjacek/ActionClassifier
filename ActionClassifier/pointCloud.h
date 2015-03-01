#pragma once

#include "ActionClassifier.h"
#include "utils.h"
#include "pca.h"

using cv::Vec;

typedef Vec<double, 3> Point3D;
typedef cv::Point Point2D;

class PointCloud{
  public:
    vector<Point3D> points;
    Point3D x_max;
	Point3D y_max;
	Point3D z_max;
    Point3D x_min;
	Point3D y_min;
	Point3D z_min;
	Point3D centroid;

	Point3D cloudSize;

	PointCloud(Mat mat);
	void normalize();
	Point3D getCentroid();
	Point3D getStds();
	pair<Point3D, Point3D> getPrincipalComponents();
	Point3D getDims();
	vector<Point3D> getExtremePoints();
	vector<Point3D> sample(int n);
	void show();

private:
    pair<Point3D, Point3D> computeExtremes();
    MatrixXd imageToMatrix();
};

