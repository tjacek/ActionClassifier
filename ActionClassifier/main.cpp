#include "ActionClassifier.h"
#include "io.h"
#include "features.h"
#include "classifiers.h"
#include "pca.h"

void addAllExtractors(Dataset * dataset){
  addLinearStdExtractor(dataset);
  //addPcaExtractor(dataset);
}

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	AddExtractorsFunc extractor= &addLinearStdExtractor;
	return  buildDataset(imageList,addAllExtractors);
}

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 string labels ="C:/Users/user/Desktop/kwolek/labels.txt";
 Categories categories=readCategories(labels);
 ImageList fullSet=getImageList(dirName);
// evaluate(dirName,labels);
 //readCategories(labels);
 //test_pca();
 Labels tlabels= getLabels(fullSet,categories);

 Dataset * dataset=getDataset(dirName);
 cout << dataset->toArff(tlabels);

 system("pause");

}