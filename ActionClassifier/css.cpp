#include "css.h"

 vector<Point>* extractCSSFeatures(Mat * image){
   vector<Point>* points=new vector<Point>();
   detectEdge(image);
   getCurve(image);
   return points;
 }

static const double minThreshold = 30.0;
static const double maxThreshold = 50.0;

void detectEdge(Mat * mat){
  cv::Canny(*mat, *mat, minThreshold, maxThreshold);
}

Curve * getCurve(Mat * image){
  vector<vector<cv::Point> > contours;
  vector<cv::Vec4i> hierarchy;

  cv::findContours(*image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
  vector<cv::Point> countur=contours[0];
  Curve * curve = new Curve(countur);
  return curve;
}

Curve::Curve(vector<cv::Point> countur){
  vector<double> x_vec;
  vector<double> y_vec;
  vector<Point>::iterator it;
  int i=0;
  for(it=countur.begin(); it!=countur.end(); ++it )
  {
     Point point=*it;
	 x_vec.push_back(point.x);
	 x_vec.push_back(point.y);
  }
  x=new Mat(x_vec);
  y=new Mat(y_vec);
}


