#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "classifiers.h"
#include "pca.h"
#include "css.h"
#include "utils.h"

void addAllExtractors(Dataset * dataset){
  addLinearStdExtractor(dataset);
  //addPcaExtractor(dataset);
}

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	AddExtractorsFunc extractor= &addLinearStdExtractor;
	return  buildDataset(imageList,addAllExtractors);
}

void testCSS(ImageList imageList){
	Images images=readImages(imageList);
	DepthImage dimage=images->at(0);
    extractCSSFeatures(&dimage.image);
	//showImages(images);
}

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 string labels ="C:/Users/user/Desktop/kwolek/labels.txt";
 //Categories categories=readCategories(labels);
 ImageList imageList=getImageList(dirName);
 testCSS( imageList);
 system("pause");
}