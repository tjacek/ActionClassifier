#include "features.h"

void addCentroidExtractor(Dataset * dataset){
  CentroidExtractor* centroidExtractor=new CentroidExtractor();
  dataset->registerExtractor(centroidExtractor);
}

int CentroidExtractor::numberOfFeatures(){
  return 2;
}

string CentroidExtractor:: featureName(int i){
  string str="centroid_" + intToString(i);
  return str;
}

FeatureVector CentroidExtractor::getFeatures(DepthImage image){
  FeatureVector features=new vector<float>();
  double x=0.0;
  double y=0.0;
  double count=0.0;
  Mat * m_image=&image.image;
  for(int i=0;i<m_image->rows;i++){
	const double* row_i = m_image->ptr<double>(i);
	for(int j = 0; j < m_image->cols; j++){
	  if(row_i[j]!=0){
		x+=i;
		y+=j;
		count+=1.0;
	   }
	}
  }

  x/=count;
  y/=count;
  cout << x << " " << y << " " << count << endl;
  features->push_back((float) x);
  features->push_back((float) y);

  return features;
};