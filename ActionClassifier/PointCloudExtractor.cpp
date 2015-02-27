#include "features.h"
#include "utils.h"
#include "pointCloud.h"

void addFeatures(FeatureVector features, Point3D point){
  for(int i=0;i<point.rows;i++){
	features->push_back(point.val[i]);
  }
}

void addPointCloudExtractor(Dataset * dataset){
  PointCloudExtractor* extractor=new PointCloudExtractor();
  dataset->registerExtractor(extractor);
}

int PointCloudExtractor::numberOfFeatures(){
  return 15;
}

string PointCloudExtractor:: featureName(int i){
  string str="point_cloud" + intToString(i);
  return str;
}

FeatureVector PointCloudExtractor::getFeatures(DepthImage image){
  FeatureVector fullVect= new vector<double>();
  PointCloud pointCloud(image.image);
  pointCloud.normalize();
  addFeatures(fullVect,pointCloud.getCentroid());
  addFeatures(fullVect,pointCloud.getStds());
  pair<Point3D,Point3D> eigenvectors=pointCloud.getPrincipalComponents();
  addFeatures(fullVect,eigenvectors.first);
  addFeatures(fullVect,eigenvectors.second);
  addFeatures(fullVect,pointCloud.getDims());
  return fullVect;
}