#include "ActionClassifier.h"
using cv::Point;

class Curve{
 public:
   Mat * x;
   Mat * y;
   Curve(vector<cv::Point> countur);
};

extern vector<Point>* extractCSSFeatures(Mat * image);
extern void detectEdge(Mat * mat);
extern Curve * getCurve(Mat * image);