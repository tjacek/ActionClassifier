#include "ActionClassifier.h"
using cv::Point;

class Curve{
 public:
   Mat * x;
   Mat * y;
   Mat * smooth_x;
   Mat * smooth_y;
   Mat  dX;
   Mat  dY;
   Mat  ddX;
   Mat  ddY;
   Mat G;
   Mat dG;
   Mat ddG;
   Curve(vector<cv::Point> countur);
   void smooth(double sigma);
   void computeDervatives();
};

extern vector<Point>* extractCSSFeatures(Mat * image);
extern void detectEdge(Mat * mat);
extern Curve * getCurve(Mat * image);
extern Mat gauss_conv(Mat in,Mat out,double sigma);
extern void draw(Mat image,vector<vector<cv::Point> >* contours,vector<cv::Vec4i>* hierarchy);
