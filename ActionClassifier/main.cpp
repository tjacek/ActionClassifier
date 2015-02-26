#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "classifiers.h"
#include "pca.h"
#include "shapeContext.h"
#include "utils.h"
#include "pointCloud.h"

void addAllExtractors(Dataset * dataset){
  addShapeContextExtractor(dataset);
  //addPcaExtractor(dataset);
}

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	return  buildDataset(imageList,addAllExtractors);
}

void testCSS(ImageList imageList){

  Images images=readImages(imageList);
  //showHistograms(images);
  DepthImage dimage=images->at(0);
  PointCloud pointCloud(dimage.image);
  pointCloud.normalize();
  pointCloud.getCentroid();
  pointCloud.getStds();
  pointCloud.getPrincipalComponents();
  //test_pca();
  //vector<Mat> im=projection(&dimage.image);
  //saveImages(im);
 // cout << getShapeContext(100,&dimage.image)->toVector().at(0);
	//showImages(images);

}
 
void createArffDataset(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 string labelsFile ="C:/Users/user/Desktop/kwolek/labels.txt";
 Categories cat=readCategories(labelsFile);
 ImageList imageList = getImageList( dirName);
 Labels labels= getLabels(imageList, cat );
 Dataset * dataset=getDataset( dirName);
 dataset->dimReduction(20);
 ofstream myfile;
 myfile.open ("shapeContext2.arff");
 myfile << dataset->toArff(labels);
 myfile.close();
// cout <<dataset->toArff(labels);
}

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 string labelsFile ="C:/Users/user/Desktop/kwolek/labels.txt";
 ImageList imageList = getImageList( dirName);

 testCSS(imageList);
// createArffDataset();
/* 
 //Categories categories=readCategories(labels);
 /*ImageList imageList=getImageList(dirName);
 Images images=readImages(imageList);
 showCounturs(images);*/
 system("pause");
}