#include "classifiers.h"
#include "io.h"
#include "utils.h"

SplitedImages::SplitedImages(ImageList imageList){
  srand( time( NULL ) );
  training=new vector<string>();
  test=new vector<string>();
  vector<string>::iterator it;
  for(it=imageList->begin(); it!=imageList->end(); ++it )
  {
	string filename= *it;
	if(biasedCoin()){
	  training->push_back(filename);
	}else{
      test->push_back(filename);
	}
  }
}

void evaluate(string imageDir,string categoryFile){
  Categories categories=readCategories(categoryFile);
  ImageList fullSet=getImageList(imageDir);
  SplitedImages splitedImages(fullSet);
  Classifier * cls=buildClassifier(splitedImages.training,categories);
  Labels  trueLabels= getLabels(splitedImages.test,categories);
  Labels predictedLabels=getPredictedLabels(splitedImages.test, cls);
  cout << "\n True "<< *trueLabels <<"\n";
  cout << "Predict "<< *predictedLabels <<"\n";
  cout << "TP "<< truePositives(trueLabels,predictedLabels)<<"\n";
}

int truePositives(Labels  trueLabels,Labels predictedLabels){
  int tp=0;
  for(int i=0;i<trueLabels->cols;i++){
	 double tl=trueLabels->data[i]; //at<double>(i,0);
	 double pl=predictedLabels->data[i]; //->at<double>(i,0);
	 if(tl==pl){
		tp++;
	 }
  }
  return tp;
}

Labels getPredictedLabels(ImageList imageList,Classifier * cls){
  vector<string>::iterator it;
  vector<float>* fullFeatures = new vector<float>();
  for(it=imageList->begin(); it!=imageList->end(); ++it )
  {
	 string filename= *it;
     DepthImage image(filename);
	 double category=cls->predict(image);
	 fullFeatures->push_back(category);
  }
  return new cv::Mat(*fullFeatures);
}

Classifier * buildClassifier(ImageList trainingSet,Categories categories){
  Classifier * classifier=new SVMClassifier();
  Labels labels= getLabels(trainingSet,categories );
  Dataset* dataset=buildDataset( trainingSet, addAllExtractors);
  classifier->learn(labels,dataset);
  return classifier;
}

Labels getLabels(ImageList imageList,Categories categories ){
  vector<string>::iterator it;
  vector<float>* fullFeatures = new vector<float>();
  for(it=imageList->begin(); it!=imageList->end(); ++it )
  {
	 string filename= *it;
	 double i=categories[filename];
	 fullFeatures->push_back(i);
  }
  return new cv::Mat(*fullFeatures);
}

Categories readCategories(string name){
  Categories categories;
  ifstream in_stream;

  string line;
  in_stream.open(name);

  while(!in_stream.eof()){
    in_stream >> line;
	string filename;
	int category=0;
	std::stringstream ss(line);
    std::string catStr;
	getline(ss, filename, ',');
	getline(ss, catStr, ',');
	istringstream iss(catStr);
	iss >> category;
	categories.insert ( std::pair<string,int>(filename,category));
  }
  return categories;
}