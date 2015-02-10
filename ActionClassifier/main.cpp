#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "pca.h"



void addAllExtractors(Dataset * dataset){
  //addLinearStdExtractor(dataset);
  addPcaExtractor(dataset);
}

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	AddExtractorsFunc extractor= &addLinearStdExtractor;
	return  buildDataset(imageList,addAllExtractors);
}

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 //test_pca();
 Dataset * dataset=getDataset(dirName);
 cout << dataset->toString();
 //ImageList files=getImageList(dirName); 
 //showImageList(files);
 system("pause");

}