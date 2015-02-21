#include "pca.h"

void test_pca(){
  MatrixXd dataPoints = vectorsToMat(generateData(200));
  cout << dataPoints;
  cout << pca(dataPoints);
}

void addPcaExtractor(Dataset * dataset){
  PcaExtractor* extractor=new PcaExtractor();
  dataset->registerExtractor(extractor);
}

int PcaExtractor::numberOfFeatures(){
  return 9;
}

string PcaExtractor:: featureName(int i){
  string str="pca_" + intToString(i);
  return str;
}

FeatureVector PcaExtractor::getFeatures(DepthImage image){
  FeatureVector features=new vector<float>();
  MatrixXd dataPoints=imageToMatrix(&image.image);
  EigenVectors eigenVectors=pca( dataPoints);
  for(int i=0;i<eigenVectors.rows();i++){
	for(int j=0;j<eigenVectors.cols();j++){
		features->push_back(eigenVectors(i,j));
	}
  }
  return features;
}

MatrixXd imageToMatrix(Mat* image){
	int size=(image->cols/100 +1) * (image->rows/100 + 1);
	//cout << "\n size" << size;
	MatrixXd data=MatrixXd::Zero(3, size);
	int k=0;
	for(int i=0;i<image->rows;i++){
	const uchar* row_i = image->ptr<uchar>(i);
	  for(int j = 0; j < image->cols; j++){
		  if( (i % 100)==0 && (j % 100)==0){
		    double x=i;
		    double y=j;
		    double z=row_i[j];
		    data.col(k) << x, y, z;
		    k++;
		  }
	  }
    }
	return data;
}

MatrixXd vectorsToMat(vector<vector<double>>  vectors){
  int height=vectors.size();
  int width=vectors.at(0).size();
  MatrixXd matrix = MatrixXd::Zero(height,width);
  for(int i=0;i<height;i++){
	 vector<double> vector= vectors.at(i);
     for(int j=0;j<width;j++){
		 matrix(i,j) =vector.at(j);
    } 
  }  
  return matrix;
}

EigenVectors pca(MatrixXd dataPoints){
  int dim = dataPoints.cols(); 
  int size = dataPoints.rows();
  
  cout << dim << " " ;
//  cout << size << " ";
  double mean; VectorXd meanVector;
  for (int i = 0; i < dim; i++){
	   mean = (dataPoints.col(i).sum())/  ((double)size);		 
	   meanVector  = VectorXd::Constant(size,mean); 
	   dataPoints.col(i) -= meanVector;
	  // std::cout << meanVector.transpose() << "\n" << DataPoints.col(i).transpose() << "\n\n";
  }
  //  cout << dataPoints;
  
  MatrixXd Covariance = MatrixXd::Zero(dim, dim);
  Covariance = (1 / (double) size)  * dataPoints.transpose()* dataPoints;
  EigenSolver<MatrixXd> m_solve(Covariance);
 
 // cout << Covariance.cols();

  MatrixXd eigenVectors = MatrixXd::Zero(dim,dim); 
  eigenVectors = m_solve.eigenvectors().real();

  VectorXd eigenvalues = VectorXd::Zero(dim);
  eigenvalues = m_solve.eigenvalues().real();

  PermutationIndices pi;
  for (int i = 0 ; i < dim; i++)
	  pi.push_back(std::make_pair(eigenvalues(i), i));

  sort(pi.begin(), pi.end());
  
  int index=pi.at(0).second;
  return eigenVectors.col(index);
}

vector<vector<double>> generateData(int n){
  vector<vector<double>> vectors;
     std::default_random_engine re;
  std::uniform_real_distribution<double> unif(0,100.0);
  std::uniform_real_distribution<double> noise(0,1.0);
  for(int i=0;i<n;i++){
	 vector<double> vector;
	 double x=unif(re);
	 double y=unif(re);
	 double z=0.0;//noise(re);
	 double t=0.0;//noise(re);
	 //cout << x << " " << y << " " << z << " " << t <<"\n";
	 vector.push_back(x);
     vector.push_back(y);
     vector.push_back(z);
     vector.push_back(t);
	 
     vectors.push_back(vector);
  }
  return vectors;
}