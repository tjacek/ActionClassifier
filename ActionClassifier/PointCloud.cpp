#include "pointCloud.h"
#include "shapeContext.h"

Point3D getPoint(double depthX,double depthY,double depthZ,double resolutionX,double resolutionY){
  double horizontalFov = 58.5;   
  double verticalFov   = 45.6;  
  double xzFactor = tan( (horizontalFov /2) * (M_PI/180)) * 2;
  double yzFactor = tan( (verticalFov /2)   * (M_PI/180)) * 2;
  double normalizedX = (depthX / resolutionX)- 0.5;
  double normalizedY = 0.5 - (depthY / resolutionY);
  Point3D point;
  point.val[0] = normalizedX * depthZ * xzFactor;
  point.val[1] = normalizedY * depthZ * yzFactor;
  point.val[2] = depthZ;
  return point;
}

PointCloud::PointCloud(Mat mat){
  for(int i=0;i<mat.rows;i++){
    for(int j=0;j<mat.cols;j++){
      double z= (double) mat.at<uchar>(i,j);
	  if(z!=255){
		  Point3D point=getPoint(i,j,z,mat.cols,mat.rows);
	    points.push_back(point);
	  }
    } 
  }
}

double PointCloud::r(){
	return L2(cloudSize);
}


Point3D PointCloud::getCentroid(){
  centroid.zeros();
  double size=(double) points.size();
  for(int i=0;i<points.size();i++){
    Point3D current=points.at(i);
	for(int j=0;j<3;j++){
		centroid.val[j]+=current.val[j];
	}
  }
  for(int j=0;j<3;j++){
	centroid.val[j]/=size;
  }  
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

void PointCloud::removeOutliers(){
  Sample sample;
  for(int i=0;i<points.size();i++){
	Point3D current=points.at(i);
	sample.push_back(current.val[3]);
  }
  double u=mean(sample);
  double threshold=5;//3*standardDeviation(sample);
  vector<int> indexs;
  for(int i=0;i<points.size();i++){
	Point3D current=points.at(i);
	if(abs(current.val[3] - u)>threshold){
	  indexs.push_back(i);
	}
  }
  for(int i=indexs.size()-1;i>=0;i--){
    int j=indexs.at(i);
	points.erase(points.begin(),points.begin()+j);
  }
}

void PointCloud::save(string name){
 /*string size=intToString(points.size());
 string header ="VERSION .7 \nFIELDS x y z \nSIZE 4 4 4 \nTYPE F F F \nWIDTH ";
 header+=size + " \nHEIGHT 1 \nVIEWPOINT 0 0 0 1 0 0 0 \nPOINTS " ;
 header+= size+ " \nDATA ascii \n";*/
 ofstream myfile;
 myfile.open (name);
 //myfile << header;
 for(int i=0;i<points.size();i++){
   Point3D p=points.at(i);
   myfile << p.val[0]<< " "<<p.val[1] << " " << p.val[2] <<"\n";
  }
  myfile.close();
}

vector<Point3D> PointCloud::getExtremePoints(){
  vector<Point3D> epoints;
  epoints.reserve(10);
  //epoints.push_back(points.at(x_min));
  /*epoints.push_back(points.at(y_min));
  epoints.push_back(points.at(z_min));*/
  //epoints.push_back(points.at(x_max));
  /*epoints.push_back(points.at(y_max));
  epoints.push_back(points.at(z_max));*/
  
  epoints.push_back(centroid);
  return epoints;
}

vector<Point3D> PointCloud::sample(int n){
  vector<Point3D> sample;
  for(int i=0;i<n;i++){
	int index= rand() % this->points.size(); 
	sample.push_back( points.at(index));
  }
  return sample;
}

void PointCloud::show(){
  for(int i=0;i<points.size();i++){
    if(i % 100 ==0 ){
     cout << points.at(i) << endl;
	}
  }
}

void PointCloud::normalize(){
//	show();
  pair<Point3D, Point3D> extremes=computeExtremes();
  Point3D min=extremes.first;
  Point3D max=extremes.second;
 // cout << min << " | " << max <<"\n";
  for(int i=0;i<points.size();i++){
    Point3D * current=&points.at(i);
     for(int j=0;j<3;j++){
	  current->val[j]-=min.val[j];
	 }
  }
  //cout << "******************************\n";
  //show();
   //cout << "min " << min <<"\n";
  cloudSize=max-min;
  //cout << cloudSize << endl;
  double r=L2(cloudSize);
  for(int i=0;i<points.size();i++){
    Point3D * current=&points.at(i);
	for(int j=0;j<3;j++){
	  current->val[j]/=r;
	  current->val[j]*=1000.0;
	}
	//cout << current << endl;
  }
  cloudSize*=1000;
  //show();
   cout << x_min;

}

pair<Point3D, Point3D> PointCloud::computeExtremes(){
  pair<Point3D, Point3D> extremes;
  Point3D minV(9999,9999,9999);
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

  x_max=(int)maxIndex.val[0];
  y_max=(int)maxIndex.val[1];
  z_max=(int)maxIndex.val[2];

  x_min=(int) minIndex.val[0];
  y_min=(int) minIndex.val[1];
  z_min=(int) minIndex.val[2];

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