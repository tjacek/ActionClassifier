#include "ActionClassifier.h"

using cv::Vec;

typedef Vec<double, 3> Point3D;

class PointCloud{
  public:
    vector<Point3D> points;
    Point3D maxValues;

	PointCloud(Mat mat);
	void normalize();

  private:
    pair<Point3D, Point3D> computeExtremes();
};