#include "ActionClassifier.h"
#include "io.h"
#include "features.h"

ImageDescriptor getImageDescriptor(DepthImage image,vector<FeatureExtractor*> extractors){
  vector<FeatureExtractor*>::iterator it;
  vector<float>* fullFeatures = new vector<float>();
  for(it=extractors.begin(); it!=extractors.end(); ++it )
  {
	FeatureVector featureVector = (*it)->getFeatures(image);
	fullFeatures->insert(fullFeatures->end(), featureVector->begin(), featureVector->end());
  }
  return cv::Mat(*fullFeatures);
}

Dataset * buildDataset(ImageList imageList, AddExtractorsFunc addExtractors){
  Dataset * dataset = new Dataset();
  addExtractors(dataset);
  vector<string>::iterator it;
  for( it=imageList->begin(); it!=imageList->end(); ++it )
  {
	  string imageName=(*it);
	  cv::Mat image = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	 // cout << image.rows << " " << image.cols ;
	  dataset->addExample(&image);
  }
  return dataset;
}

 void Dataset::addExample(DepthImage image){
   examples.push_back(getImageDescriptor(image,extractors));
 }

void Dataset::registerExtractor(FeatureExtractor* extractor){
	extractors.push_back(extractor);
}

Mat Dataset::toMat(){
  int size=examples.size();
  int dim=examples[0].rows;
  Mat mat2D(size,dim,CV_LOAD_IMAGE_GRAYSCALE);
  vector<ImageDescriptor>::iterator it;
  int i=0;
  for(it=examples.begin(); it!=examples.end(); ++it )
  {   
	  Mat mat1D=*it;
	  mat2D.row(i)=mat1D.row(0);
	  i++;
  }
  return mat2D;
}

string  matToString(Mat mat){
  string s="";
  cv::Size size = mat.size();
  for(int i=0;i<size.height;i++){
	int raw= mat.data[i];
	string tmp; 
    sprintf((char*)tmp.c_str(), "%d", raw);
    string str2 = tmp.c_str();
	s+= str2 +",";
  }
  return s;
}

string Dataset::toString(){
  string str="";
  vector<ImageDescriptor>::iterator it;
  for(it=examples.begin(); it!=examples.end(); ++it )
  {
     ImageDescriptor features=*it;
     string buf = matToString(features);
	 str+=buf + "\n";
  }
  return str;
}

