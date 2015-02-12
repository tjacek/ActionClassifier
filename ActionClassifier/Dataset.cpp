#include "ActionClassifier.h"
#include "io.h"
#include "features.h"

DepthImage::DepthImage(string imageName){
  this->name=imageName;
  this->image = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
}

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
	  DepthImage image(imageName);
	  dataset->addExample(image);
  }
  return dataset;
}

 void Dataset::addExample(DepthImage image){
   examples.push_back(getImageDescriptor(image,extractors));
 }

void Dataset::registerExtractor(FeatureExtractor* extractor){
	extractors.push_back(extractor);
}

Mat* Dataset::toMat(){
  /*examples.size;
  cv::Mat matAngles(examples.size, examples.at(0).size,CV_LOAD_IMAGE_GRAYSCALE);
  for(int i=0; i<matAngles.rows; ++i){
    for(int j=0; j<matAngles.cols; ++j){
       
      matAngles.at<uchar>(i) = examples.at(i);
    }
  }*/
  int size=examples.size();
  int dim=examples[0].rows;
  Mat * mat2D=new Mat(size,dim,CV_32F);
  vector<ImageDescriptor>::iterator it;
  int i=0;
  for(it=examples.begin(); it!=examples.end(); ++it )
  {   
	  Mat mat1D=*it;
	  mat2D->row(i)=mat1D.row(0);
	  i++;
  }
  //cout << &mat2D;
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

