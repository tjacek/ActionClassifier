#include "ActionClassifier.h"
#include "io.h"

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	return  buildDataset(imageList,NULL);
}

Dataset * buildDataset(ImageList imageList, AddExtractorsFunc addExtractors){
  Dataset * dataset = new Dataset();
  addExtractors(dataset);
  vector<string>::iterator it;
  for( it=imageList->begin(); it!=imageList->end(); ++it )
  {
	  string imageName=(*it);
	  cv::Mat image = cv::imread(imageName, CV_LOAD_IMAGE_COLOR); 
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