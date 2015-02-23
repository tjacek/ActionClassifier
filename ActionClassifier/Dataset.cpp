#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "pca.h"

DepthImage::DepthImage(string imageName){
  this->name=imageName;
  this->image = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
}

vector<double> getImageDescriptor(DepthImage image,vector<FeatureExtractor*> extractors){
  vector<FeatureExtractor*>::iterator it;
  vector<double>* fullFeatures = new vector<double>();
  for(it=extractors.begin(); it!=extractors.end(); ++it )
  {
	FeatureVector featureVector = (*it)->getFeatures(image);
	fullFeatures->insert(fullFeatures->end(), featureVector->begin(), featureVector->end());
  }
  return *fullFeatures;
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
   desc->push_back(getImageDescriptor(image,extractors));
 }

void Dataset::registerExtractor(FeatureExtractor* extractor){
	extractors.push_back(extractor);
}

int Dataset::numberOfFeatures(){
  vector<FeatureExtractor*>::iterator it;
  int counter=0;
  for(it=extractors.begin(); it!=extractors.end(); ++it )
  {
	counter+= (*it)->numberOfFeatures();
  }
  return counter;
}

Mat* Dataset::toMat(){
//  int size=examples->size();
  //int dim=examples->at(0).rows();
  Mat * mat2D=new Mat(8,20,CV_32F);
  /*vector<ImageDescriptor>::iterator it;
  int i=0;
  for(it=examples->begin(); it!=examples->end(); ++it )
  {   
	  Mat mat1D=*it;
	  mat2D->row(i)=mat1D.row(0);
	  i++;
  }*/
  return mat2D;
}

string  matToString(Mat mat){
  string s="";
  cv::Size size = mat.size();
  for(int i=0;i<size.height;i++){
	  float raw= mat.at<float>(i,0);
	string tmp; 
    sprintf((char*)tmp.c_str(), "%f", raw);
    string str2 = tmp.c_str();
	if(i==size.height-1){
	  s+= str2 ;
	}else{
      s+=str2+",";
	}
  }
  return s;
}

Dataset::Dataset(){
  desc=new  vector<vector<double>>;
//  examples=new vector<ImageDescriptor>();
}

void Dataset::dimReduction(int k){
  vector<vector<double>> * old=desc;
  this->desc=new vector<vector<double>>();
  int size=old->size();
  MatrixXd pca_projc=pca(20,vectorsToMat(*old));
  cout << "\n & "<< size <<"\n"; 
  for(int i=0;i<size;i++){
    vector<double> point=old->at(i);
	vector<double> newPoint= applyProjection(point,pca_projc);
	desc->push_back(newPoint);
  }
}

string Dataset::toArff(Labels labels){
  string arff="@RELATION DepthMaps \n";
  arff+=getAttributes();
  arff+="@attribute class numeric";
  arff+="\n @DATA \n";
  arff+=getData(labels);
  return arff;
}

string Dataset::toString(){
  string str="";
  vector<ImageDescriptor>::iterator it;
  /*for(it=examples->begin(); it!=examples->end(); ++it )
  {
     ImageDescriptor features=*it;
     string buf = matToString(features);
	 str+=buf + "\n";
  }*/
  return str;
}

string  Dataset::getAttributes(){
  string str="";
  vector<FeatureExtractor*>::iterator it;
  vector<float>* fullFeatures = new vector<float>();
  for(it=extractors.begin(); it!=extractors.end(); ++it )
  {
	int size=(*it)->numberOfFeatures();
	for(int i=0;i<size;i++){
	  string feature=(*it)->featureName(i);
	  str+="@ATTRIBUTE " + feature +" numeric\n";
	}
  }
  return str;
}

string Dataset::getData(Labels labels){
  string str="";
  vector<ImageDescriptor>::iterator it;
  cout << desc->size() << " \n";
  for(int i=0;i<desc->size();i++){
    string line="";
    vector<double> v=desc->at(i);
	for(int j=0;j<v.size();j++){
      double raw=v.at(j);
      string tmp; 
      sprintf((char*)tmp.c_str(), "%f", raw);
      string str2 = tmp.c_str();
      line+=str2+",";
	}
    int category=(int) labels->at<float>(i,0);
    line+=intToString(category)+"\n";
	str+=line;
  }
  /*for(it=examples->begin(); it!=examples->end(); ++it )
  {
     ImageDescriptor features=*it;
	 int category=(int) labels->at<float>(i,0);
	 string value= matToString(features);
     string buf = value+","+intToString(category) +"\n" ;
	 str+=buf;
	 i++;
  }*/
  return str;
}

vector<DepthImage>* readImages(ImageList imageNames){
  vector<string>::iterator it;
  vector<DepthImage>* dimages=new vector<DepthImage>();
  for(it=imageNames->begin(); it!=imageNames->end(); ++it )
  {
    string name=*it;
	DepthImage image(name);
	dimages->push_back(image);
  }
  return dimages;
}

void showImages(vector<DepthImage> * images){
  vector<DepthImage>::iterator it;
  int i=0;
  for(it=images->begin(); it!=images->end(); ++it )
  {
     DepthImage image=*it;
	 cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	 cv::imshow( "Display window", image.image );                  
     cv::waitKey(0);
  }
}
