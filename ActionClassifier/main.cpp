#include "ActionClassifier.h"
#include "io.h"
#include "features.h"

Dataset * getDataset(string dirName){
	ImageList imageList = getImageList( dirName);
	AddExtractorsFunc extractor= &addLinearStdExtractor;
	return  buildDataset(imageList,extractor);
}

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 getDataset(dirName);
 //ImageList files=getImageList(dirName); 
 //showImageList(files);
 system("pause");

}