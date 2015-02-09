#include "features.h"

void addCentroidExtractor(Dataset * dataset){
  CentroidExtractor* centroidExtractor=new CentroidExtractor();
  dataset->registerExtractor(centroidExtractor);
}

FeatureVector CentroidExtractor::getFeatures(DepthImage image){
  FeatureVector features=new vector<float>();
  double x=0.0;
  double y=0.0;
  double count=0.0;

  for(int i=0;i<image->rows;i++){
	const double* row_i = image->ptr<double>(i);
	for(int j = 0; j < image->cols; j++){
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