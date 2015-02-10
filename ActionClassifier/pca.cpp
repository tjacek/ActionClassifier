#include "pca.h"

void test_pca(){
  MatrixXd dataPoints = MatrixXd::Random(5, 10);
  //cout <<
  pca(dataPoints);
}

void addPcaExtractor(Dataset * dataset){
  PcaExtractor* extractor=new PcaExtractor();
  dataset->registerExtractor(extractor);
}

FeatureVector PcaExtractor::getFeatures(DepthImage image){
  FeatureVector features=new vector<float>();
  MatrixXd dataPoints=imageToMatrix(image);
  EigenVectors eigenVectors=pca( dataPoints);
 // cout << eigenVectors << "End of eigenvector";
  for(int i=0;i<eigenVectors.rows();i++){
    //MatrixXd eigenvector=eigenVectors.row(i);
	for(int j=0;j<eigenVectors.cols();j++){
		features->push_back(eigenVectors(i,j));
	}
  }
  return features;
}

MatrixXd imageToMatrix(DepthImage image){
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
	//cout << " " << k;
	//cout << data;
	return data;
}

EigenVectors pca(MatrixXd dataPoints){
 // unsigned int startDim = 3;
  unsigned int startDim = dataPoints.cols();
  unsigned int size = dataPoints.rows(); 
  double mean; VectorXd meanVector;
  for (int i = 0; i < size; i++){
	   mean = (dataPoints.row(i).sum())/startDim;		 //compute mean
	   meanVector  = VectorXd::Constant(startDim,mean); // create a vector with constant value = mean
	   dataPoints.row(i) -= meanVector;
	  // std::cout << meanVector.transpose() << "\n" << DataPoints.col(i).transpose() << "\n\n";
  }
  //  cout << dataPoints;

  MatrixXd Covariance = MatrixXd::Zero(startDim, startDim);
  Covariance = (1 / (double) size) * dataPoints * dataPoints.transpose();
  EigenSolver<MatrixXd> m_solve(Covariance);

  MatrixXd eigenVectors = MatrixXd::Zero(size, startDim);  // matrix (n x m) (points, dims)
  eigenVectors = m_solve.eigenvectors().real();
  return eigenVectors;
}
