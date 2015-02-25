#include "pca.h"

void addPcaExtractor(Dataset * dataset){
  PcaExtractor* extractor=new PcaExtractor();
  dataset->registerExtractor(extractor);
}

int PcaExtractor::numberOfFeatures(){
  return 9;
}

string PcaExtractor:: featureName(int i){
  string str="pca_" + intToString(i);
  return str;
}

FeatureVector PcaExtractor::getFeatures(DepthImage image){
  FeatureVector features=new vector<double>();
  /*MatrixXd dataPoints=imageToMatrix(&image.image);
  EigenVectors eigenVectors=pca(3,dataPoints);
  for(int i=0;i<eigenVectors.rows();i++){
	for(int j=0;j<eigenVectors.cols();j++){
		features->push_back(eigenVectors(i,j));
	}
  }*/
  return features;
}