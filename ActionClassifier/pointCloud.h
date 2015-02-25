#include "ActionClassifier.h"
#include "utils.h"
#include "pca.h"

using cv::Vec;

typedef Vec<double, 3> Point3D;

class PointCloud{
  public:
    vector<Point3D> points;
    Point3D maxValues;

	PointCloud(Mat mat);
	void normalize();
	Point3D getCentroid();
	Point3D getStds();
	pair<Point3D, Point3D> getPrincipalComponents();

private:
    pair<Point3D, Point3D> computeExtremes();
    MatrixXd imageToMatrix();
};

