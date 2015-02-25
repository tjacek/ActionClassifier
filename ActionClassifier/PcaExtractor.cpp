#include "pca.h"

void addPcaExtractor(Dataset * dataset){
  PcaExtractor* extractor=new PcaExtractor();
  dataset->registerExtractor(extractor);
}

MatrixXd imageToMatrix(Mat* image){
	int size=(image->cols/100 +1) * (image->rows/100 + 1);
	//cout << "\n size" << size;
	MatrixXd data=MatrixXd::Zero(3, size);
	int k=0;
	for(int i=0;i<image->rows;i++){
	const uchar* row_i = image->ptr<uchar>(i);
	  for(int j = 0; j < image->cols; j++){
		  if( (i % 100)==0 && (j % 100)==0){
		    double x=i;
		    double y=j;
		    double z=row_i[j];
		    data.col(k) << x, y, z;
		    k++;
		  }
	  }
    }
	return data;
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