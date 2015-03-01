#pragma once
#define _USE_MATH_DEFINES

#include "ActionClassifier.h"
#include "PointCloud.h"
#include "utils.h"

using cv::Point;

typedef cv::Point* PolarVector;
typedef vector<cv::Point> Points;

class OnlineHistogram{
  public:
	int dimR;
	int dimTheta;
	double ** bins;
	double maxR;
    double maxTheta;

	OnlineHistogram(int dimR,int dimTheta,double maxR=800);
	void addToHistogram(double r_i,double theta_i);
	void normalize();
	void show();
    FeatureVector toVector();

  private:
    double rStep;
	double thetaStep;
};

class Histogram3D{
  public:
    double *** bins;
    int size;
    Point3D maxValues;
  
    Histogram3D(double r);
    void addToHistogram(double ksi,double theta,double psi);
    void normalize();
    void show();
    FeatureVector toVector();
};

extern OnlineHistogram * getShapeContext(int n,Mat * image);
extern Points samplePoints(int n,Mat * image);
extern PolarVector getPolarVector(Point p1,Point p2);

extern Histogram3D * getShapeContext3D(int n,PointCloud point);
extern void addPoints(Point3D point,PointCloud pointCloud,Histogram3D * histogram);
extern double L2(Point3D point);