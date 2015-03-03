#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "classifiers.h"
#include "pca.h"
#include "shapeContext.h"
#include "utils.h"
#include "pointCloud.h"
#include "transform.h"

void addAllExtractors(Dataset * dataset){
  addShapeContext3DExtractor(dataset);
  //addPcaExtractor(dataset);
}

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	return  buildDataset(imageList,addAllExtractors);
}

void fullFeatures(){
  string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
  string labelsFile ="C:/Users/user/Desktop/kwolek/labels.txt";

  Categories cat=readCategories(labelsFile);
 ImageList imageList = getImageList( dirName);
 Labels labels= getLabels(imageList, cat );
 Dataset * data1=buildDataset(imageList,addShapeContext3DExtractor);
 data1->dimReduction(20);

 //Dataset * data2=buildDataset(imageList, addPointCloudExtractor);
 //Dataset * data3=new Dataset(data1,data2);

 ofstream myfile;
 myfile.open ("shapeContext3D.arff");
 myfile << data1->toArff(labels);
 myfile.close();
}

void testCSS(ImageList imageList){

  Images images=readImages(imageList);
  //showHistograms(images);
  DepthImage dimage=images->at(0);
      Mat newImage=medianaFilter(&dimage.image);
  showImage(&newImage,"OK");
  PointCloud pointCloud(newImage);
  getShapeContext3D(200, pointCloud);
  //pointCloud.show();
  //pointCloud.normalize();
  /*pointCloud.getCentroid();
  pointCloud.getStds();
  pointCloud.getPrincipalComponents();*/
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
	//fullFeatures();
 string dirName ="C:/Users/user/Desktop/kwolek/dataset";
 ImageList imageList = getImageList( dirName);
 string labelsFile ="C:/Users/user/Desktop/kwolek/labels.txt";
 testCSS(imageList);

 //createArffDataset();
/* 
 //Categories categories=readCategories(labels);
 /*ImageList imageList=getImageList(dirName);
 Images images=readImages(imageList);
 showCounturs(images);*/
 system("pause");
}