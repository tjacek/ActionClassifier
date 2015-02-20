#define _USE_MATH_DEFINES

#include "ActionClassifier.h"
#include "css.h"

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
 
  private:
    double rStep;
	double thetaStep;
};

extern OnlineHistogram * getShapeContext(int n,Mat * image);
extern Points samplePoints(int n,Mat * image);
extern PolarVector getPolarVector(Point p1,Point p2);