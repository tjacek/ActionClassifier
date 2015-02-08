#include "ActionClassifier.h"
#include "io.h"

int main(){
 string dirName ="C:/Users/user/Desktop/kwolek/dataset"; 
 ImageList files=getImageList(dirName); 
// getFilesList(stringToTCHAR(inputFolderPath),files);
 showImageList(files);
 system("pause");

}