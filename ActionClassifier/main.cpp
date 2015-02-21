#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "classifiers.h"
#include "pca.h"
#include "shapeContext.h"
#include "utils.h"

void addAllExtractors(Dataset * dataset){
  addShapeContextExtractor(dataset);
  //addPcaExtractor(dataset);
}

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	AddExtractorsFunc extractor= &addLinearStdExtractor;
	return  buildDataset(imageList,addAllExtractors);
}

void testCSS(ImageList imageList){
  //Images images=readImages(imageList);
  //DepthImage dimage=images->at(0);
  //test_pca();
  //vector<Mat> im=projection(&dimage.image);
  //saveImages(im);
 // cout << getShapeContext(100,&dimage.image)->toVector().at(0);
	//showImages(images);
}

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 string labels ="C:/Users/user/Desktop/kwolek/labels.txt";
 Dataset * dataset=getDataset( dirName);

 //Categories categories=readCategories(labels);
 //ImageList imageList=getImageList(dirName);
 //testCSS( imageList);
 system("pause");
}