#include "classifiers.h"
#include "io.h"



void evaluate(string imageDir,string categoryFile){
  Categories categories=readCategories(categoryFile);
  ImageList trainingSet=getImageList(imageDir);
  Classifier * cls=buildClassifier(trainingSet,categories);
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
	 int i=categories[filename];
     cout << i <<"\n";
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
	cout << filename << " ^^^" <<category << "\n";
	categories.insert ( std::pair<string,int>(filename,category));
  }
  return categories;
}