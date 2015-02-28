#include "pointCloud.h"

PointCloud::PointCloud(Mat mat){
  for(int i=0;i<mat.rows;i++){
    for(int j=0;j<mat.cols;j++){
      double z= (double) mat.at<uchar>(i,j);
	  if(z!=255){
	    Point3D point3D(i,j,z);
	    points.push_back(point3D);
	  }
    } 
  }
}

Point3D PointCloud::getCentroid(){
  Point3D centroid(0,0,0);
  double size=(double) points.size();
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	centroid+=current;
  }
  centroid/=size;
  return centroid;
}

pair<Point3D, Point3D> PointCloud::getPrincipalComponents(){
  pair<Point3D, Point3D> pair;
  MatrixXd x=imageToMatrix();
  MatrixXd eigen=pca(2,x);
  for(int j=0;j<3;j++){
	pair.first.val[j]=eigen(0,j);
	pair.second.val[j]=eigen(1,j);
  }
  return pair;
}

Point3D PointCloud::getStds(){
  Point3D stds(0,0,0);
  Sample x;
  Sample y;
  Sample z;
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	x.push_back(current.val[0]);
	y.push_back(current.val[1]);
	z.push_back(current.val[2]);
  }
  stds.val[0]=standardDeviation(x);
  stds.val[1]=standardDeviation(y);
  stds.val[2]=standardDeviation(z);
  return stds;
}

Point3D PointCloud::getDims(){
  double r=0;
 
  for(int i=0;i<3;i++){
	  r+=cloudSize.val[i] *cloudSize.val[i];
  }
  r=sqrt(r);
  cout <<  r;
  for(int i=0;i<3;i++){
	  cloudSize.val[i]/=r;
  }
  return cloudSize;
}

void PointCloud::show(){
  for(int i=0;i<points.size();i++){
    cout << points.at(i);
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
  cloudSize=max-min;
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	for(int j=0;j<3;j++){
	  current.val[j]/=cloudSize.val[j];
	}
  }
}

pair<Point3D, Point3D> PointCloud::computeExtremes(){
  pair<Point3D, Point3D> extremes;
  Point3D minV(0,0,0);
  Point3D maxV(0,0,0);
  Vec<int, 3> minIndex(0,0,0);
  Vec<int, 3> maxIndex(0,0,0);

  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	for(int j=0;j<3;j++){
	  if(current.val[j]< minV.val[j]){
		minV.val[j]=current.val[j];
		minIndex.val[j]=j;
	  }
	}
	for(int j=0;j<3;j++){
	  if(current.val[j]> maxV.val[j]){
		maxV.val[j]=current.val[j];
		minIndex.val[j]=j;
	  }
	}
  }

  x_max=points.at(maxIndex.val[0]);
  y_max=points.at(maxIndex.val[1]);
  y_max=points.at(maxIndex.val[2]);

  x_min=points.at(minIndex.val[0]);
  y_min=points.at(minIndex.val[1]);
  y_min=points.at(minIndex.val[2]);

  extremes.first =minV;
  extremes.second=maxV;
  return extremes;
}

MatrixXd PointCloud::imageToMatrix(){
  MatrixXd data=MatrixXd::Zero(3, points.size());
  for(int i=0;i<points.size();i++){	
    Point3D current=points.at(i);
	for(int j=0;j<3;j++){	 
	  data(j,i)=current.val[j];
	}
  }
  return data;
}