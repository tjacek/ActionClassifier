#pragma once

#include "ActionClassifier.h"
#include "utils.h"
#include "pca.h"

using cv::Vec;

typedef cv::Point Point2D;


class PointCloud{
  public:
    vector<Point3D> points;
    int  x_max;
	int  y_max;
	int  z_max;
    int  x_min;
	int  y_min;
	int  z_min;
	Point3D centroid;

	Point3D cloudSize;

	PointCloud(Mat mat);
	void normalize();
	Point3D getCentroid();
	Point3D getStds();
	pair<Point3D, Point3D> getPrincipalComponents();
	Point3D getDims();
	void removeOutliers();
	void save(string name);
	vector<Point3D > getExtremePoints();
	vector<Point3D> sample(int n);
		double r();

	void show();

private:
    pair<Point3D, Point3D> computeExtremes();
    MatrixXd imageToMatrix();
};