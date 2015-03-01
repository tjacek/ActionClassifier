#include "shapeContext.h"

Histogram3D * getShapeContext3D(int n,PointCloud pointCloud){
  vector<Point3D> points=pointCloud.getExtremePoints();
  Histogram3D * histogram;
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	vector<Point3D> points=pointCloud.sample(n);
	addPoints(current, points, histogram);
  }
  histogram->normalize();
  return histogram;
}

void addPoints(Point3D centre,vector<Point3D> points,Histogram3D * histogram){
  for(int i=0;i<points.size();i++){
    Point3D rawpoint=points.at(i);
    Point3D point= rawpoint - centre;

	double ksi=log(L2(point));
	double theta=atan2(point.val[1],point.val[0]) + M_PI;
	double x=point.val[0];
	double y=point.val[1];
	double beta=atan2(point.val[3],sqrt(x*x+y*y)) + M_PI;
	histogram->addToHistogram(ksi,theta,beta);
  }
}

double L2(Point3D point){
  double x=point.val[0];
  double y=point.val[1];
  double z=point.val[2];
  return sqrt(x*x+y*y+z*z);
}

Histogram3D::Histogram3D(double r){
  const int size=4;
  maxValues.val[0]=log(r);
  maxValues.val[1]=2*M_PI;
  maxValues.val[2]=2*M_PI;
  bins=new double **[size];
  for(int i=0;i<size;i++){
	bins[i]=new double *[size];
	for(int j=0;j<size;j++){
      bins[i][j]=new double[size];
      for(int k=0;k<size;k++){
        bins[i][j][k]=0; 
	  }
	}
  }
}

int getIndex(double value,double size,double max){
  double step=max/size;
  return floor(value/step);
}

void Histogram3D::addToHistogram(double ksi,double theta,double psi){
  int i=getIndex(ksi,size,maxValues.val[0]);
  int j=getIndex(theta,size,maxValues.val[1]);
  int k=getIndex(psi,size,maxValues.val[2]);
  bins[i][j][k]+=1.0;
}

void Histogram3D::normalize(){
  double normalizeConst=0.0;
  for(int i=0;i<size;i++){
	for(int j=0;j<size;j++){
      for(int k=0;k<size;k++){
        normalizeConst+=bins[i][j][k];
	  }
	}
  }
  for(int i=0;i<size;i++){
	for(int j=0;j<size;j++){
      for(int k=0;k<size;k++){
        normalizeConst+=bins[i][j][k];
	  }
	}
  }
}

FeatureVector Histogram3D::toVector(){
  FeatureVector vect= new vector<double>();
  for(int i=0;i<size;i++){
	for(int j=0;j<size;j++){
	  for(int k=0;k<size;k++){
		vect->push_back(bins[i][j][k]); 
	  }
	}
  }
  return vect;
}