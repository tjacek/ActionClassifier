#include "io.h"

ImageList getImageList(string dirName){
  ImageList imageList=new vector<string>();
  getFilesList(stringToTCHAR(dirName),imageList);
  appendFullPath(imageList,dirName);
  showImageList(imageList);
  return imageList;
}

void appendFullPath(ImageList imageList, string dirName){
  vector<string>::iterator it;
  int k=0;
  for( it=imageList->begin(); it!=imageList->end(); ++it )
  {
	 string oldName= *it;
	 string newName= dirName+ "/" + oldName;
	 (*imageList)[k] =newName;
	 k++;
  }
}

TCHAR * stringToTCHAR(string input){
  wstring * wtmp =new wstring();
  for(int i = 0; i < input.length(); ++i)
  *wtmp += wchar_t( input[i] );

  const wchar_t* output = wtmp->c_str();
  return (TCHAR*) output;
}

string * WCHARToString(WCHAR * wc){
    //convert from wide char to narrow char array
    char ch[260];
    char DefChar = ' ';
    WideCharToMultiByte(CP_ACP,0,wc,-1, ch,260,&DefChar, NULL);
    
    //A std:string  using the char* constructor.
    return new string(ch);
}

void showImageList(ImageList imageList){
  vector<string>::iterator it;
  for( it=imageList->begin(); it!=imageList->end(); ++it )
  {
     cout<< *it <<'\n';
  }
}


void DisplayErrorBox(LPTSTR lpszFunction) 
{ 
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError(); 

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );

    // Display the error message and clean up

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
        (lstrlen((LPCTSTR)lpMsgBuf)+lstrlen((LPCTSTR)lpszFunction)+40)*sizeof(TCHAR)); 
    StringCchPrintf((LPTSTR)lpDisplayBuf, 
        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
        TEXT("%s failed with error %d: %s"), 
        lpszFunction, dw, lpMsgBuf); 
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK); 

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
}

int getFilesList(TCHAR * directory,ImageList  files)
{
 WIN32_FIND_DATA ffd;
   LARGE_INTEGER filesize;
   TCHAR szDir[MAX_PATH];
   size_t length_of_arg;
   HANDLE hFind = INVALID_HANDLE_VALUE;
   DWORD dwError=0;
   
   // If the directory is not specified as a command-line argument,
   // print usage.



   // Check that the input path plus 3 is not longer than MAX_PATH.
   // Three characters are for the "\*" plus NULL appended below.

   StringCchLength(directory, MAX_PATH, &length_of_arg);

   if (length_of_arg > (MAX_PATH - 3))
   {
      _tprintf(TEXT("\nDirectory path is too long.\n"));
      return -1;
   }

   _tprintf(TEXT("\nDataset directory is %s\n\n"), directory);

   // Prepare string for use with FindFile functions.  First, copy the
   // string to a buffer, then append '\*' to the directory name.

   StringCchCopy(szDir, MAX_PATH,directory);
   StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

   // Find the first file in the directory.

   hFind = FindFirstFile(szDir, &ffd);

   if (INVALID_HANDLE_VALUE == hFind) 
   {
      DisplayErrorBox(TEXT("FindFirstFile"));
      return dwError;
   } 
   
   // List all the files in the directory with some info about them.

   do
   {
      WCHAR * filename=ffd.cFileName;
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
      {
        // _tprintf(TEXT("T  %s   <DIR>\n"), filename);
      }
      else
      {
         filesize.LowPart = ffd.nFileSizeLow;
         filesize.HighPart = ffd.nFileSizeHigh;
		 string * str =WCHARToString(filename);
		 files->push_back(*str);
         //_tprintf(TEXT("  %s   %ld bytes\n"), filename, filesize.QuadPart);
      }
   }
   while (FindNextFile(hFind, &ffd) != 0);
 
   dwError = GetLastError();
   if (dwError != ERROR_NO_MORE_FILES) 
   {
      DisplayErrorBox(TEXT("FindFirstFile"));
   }

   FindClose(hFind);
   return dwError;
 }