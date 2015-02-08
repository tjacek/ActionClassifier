#include "ActionClassifier.h"

extern ImageList getImageList(string dirName);
extern void appendFullPath(ImageList imageList, string dirName);
extern int getFilesList(TCHAR * directory,ImageList  files);
extern TCHAR * stringToTCHAR(string input);
extern void DisplayErrorBox(LPTSTR lpszFunction);
extern string * WCHARToString(WCHAR * wc);
extern void showImageList(ImageList imageList);