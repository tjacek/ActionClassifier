#include "shapeContext.h"

OnlineHistogram * getShapeContext(int n,Mat * image){
  detectEdge( image);
  Points points=samplePoints( n, image);
  int size=points.size();
  OnlineHistogram * histogram=new OnlineHistogram(10,8,10);
  for(int i=0;i<size;i++){
    for(int j=i+1;j<size;j++){
      Point point1= points.at(i);
	  Point point2= points.at(j);
	  PolarVector  polarVector=getPolarVector(point1,point2);
	  histogram->addToHistogram(log(polarVector->x),polarVector->y);
	  delete polarVector;
    }
  }
  histogram->normalize();
  histogram->show();
  return histogram;
}

Points samplePoints(int n,Mat * image){
  vector<vector<cv::Point> > contours;
  vector<cv::Vec4i> hierarchy;
  cv::findContours(*image, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  vector<cv::Point> countur=contours.at(0);
  if(countur.size()<=n){
	  return countur;
  }
  vector<cv::Point> points;
  for(int i=0;i<n;i++){
    int rn=rand() % countur.size();
	points.push_back(countur.at(rn));
  }
  return points;
}

PolarVector getPolarVector(cv::Point p1,cv::Point p2){
  PolarVector polarVector= new Point();
  double x=p1.x - p2.x;
  double y=p1.y - p2.y;
  polarVector->x=sqrt(x*x + y*y);
  polarVector->y= acos(x/polarVector->x);
  return polarVector;
}

OnlineHistogram::OnlineHistogram(int dimR,int dimTheta,double maxR){
  this->maxR=maxR;
  this->maxTheta=2*M_PI;
  this->rStep= maxR /( (double) dimR);
  this->thetaStep= maxTheta /( (double) dimTheta);
  this->dimR=dimR;
  this->dimTheta=dimTheta;
  this->bins=new double*[dimR];
  for(int i=0;i<dimR;i++){
	bins[i]=new double[dimTheta];
	for(int j=0;j<dimTheta;j++){
      bins[i][j]=0; 
	}
  }
}

void OnlineHistogram::addToHistogram(double r_i,double theta_i){
  cout << rStep << "\n";
  for(int i=0;i<dimR;i++){
	if(r_i < rStep * ((double)i) ){
      for(int j=0;j<dimTheta;j++){
        if(theta_i < thetaStep * ((double)j) ){
		  bins[i][j]+=1.0;
		  return;
		}
	  }
	}
  }
}

void OnlineHistogram::normalize(){
  double normalizeConst=0.0;
  for(int i=0;i<dimR;i++){
	for(int j=0;j<dimTheta;j++){
      normalizeConst+=bins[i][j]; 
	}
  }
  for(int i=0;i<dimR;i++){
	for(int j=0;j<dimTheta;j++){
      bins[i][j]/=normalizeConst; 
	}
  }
}

void OnlineHistogram::show(){
  for(int i=0;i<dimR;i++){
	for(int j=0;j<dimTheta;j++){
       cout << bins[i][j] << " "; 
	}
	cout << "\n";
  }
}
