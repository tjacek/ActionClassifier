#include "ActionClassifier.h"
#include "io.h"
#include "features.h"

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
   vector<FeatureExtractor*>::iterator it;
   FeatureVector fullFeatures = new vector<float>();
   for(it=extractors.begin(); it!=extractors.end(); ++it )
   {
	 FeatureVector featureVector = (*it)->getFeatures(image);
	 fullFeatures->insert(fullFeatures->end(), featureVector->begin(), featureVector->end());
   }
   examples.push_back(fullFeatures);
 }

void Dataset::registerExtractor(FeatureExtractor* extractor){
	extractors.push_back(extractor);
}

string Dataset::toString(){
  string str="";
  vector<FeatureVector>::iterator it;
  for(it=examples.begin(); it!=examples.end(); ++it )
  {
     FeatureVector features=*it;
	 vector<float>::iterator it;

	 for(it=features->begin(); it!=features->end(); ++it )
     {
       float atr=*it;
	   str+= to_string(atr) + ",";
     }
	 str+="\n";
  }
  return str;
}

