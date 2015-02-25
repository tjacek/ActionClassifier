#include "pointCloud.h"

PointCloud::PointCloud(Mat mat){
  for(int i=0;i<mat.rows;i++){
    for(int j=0;j<mat.cols;j++){
      double z= (double) mat.at<uchar>(i,j);
	  Point3D point3D(i,j,z);
	  points.push_back(point3D);
    } 
  }
}

void PointCloud::normalize(){
  pair<Point3D, Point3D> extremes=computeExtremes();
  Point3D min=extremes.first;
  Point3D max=extremes.second;
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	current-=min;
  }
  maxValues=max-min;
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	for(int j=0;j<3;j++){
	  current.val[j]/=maxValues.val[j];
	}
  }
}

pair<Point3D, Point3D> PointCloud::computeExtremes(){
  pair<Point3D, Point3D> extremes;
  Point3D min;
  Point3D max;
  min.zeros();
  max.zeros();
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	for(int j=0;j<3;j++){
	  if(current.val[j]< min.val[j]){
		min.val[j]=current.val[j];
	  }
	}
	for(int j=0;j<3;j++){
	  if(current.val[j]> max.val[j]){
		max.val[j]=current.val[j];
	  }
	}
  }
  extremes.first=min;
  extremes.second=min;
  return extremes;
}