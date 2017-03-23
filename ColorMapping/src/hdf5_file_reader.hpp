
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <string>

#include "H5Cpp.h"

using namespace std;
using namespace Eigen;


class HDF5FileReader
{
public:
	HDF5FileReader(string &filename, string &datasetname) : filename_(filename), datasetname_(datasetname) {}

	void getData(void* databuf);
	void getMatrix(MatrixXd &mat);
	void getVector(VectorXd &vec);

	void setFilename(string &filename) { filename_ = filename; }
	void setDatasetname(string &datasetname) { datasetname_ = datasetname; }

	HDF5FileReader() {}
	~HDF5FileReader() {}

private:
	string filename_;
	string datasetname_;

};


void HDF5FileReader::getData(void* databuf)
{

	H5::H5File h5FileProcessor(filename_, H5F_ACC_RDONLY);
	H5::DataSet dataset = h5FileProcessor.openDataSet(datasetname_);

	H5::DataType dataType = dataset.getDataType();
	dataset.read(databuf, dataType);
}

void HDF5FileReader::getMatrix(MatrixXd &mat)
{
	int row = mat.rows();
	int col = mat.cols();
	void *databuf = malloc(sizeof(double) * 16);

	this->getData(databuf);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			mat(i, j) = static_cast<double *>(databuf)[col * i + j];
	
	free(databuf);
}

void HDF5FileReader::getVector(VectorXd &vec)
{
	void *databuf = new double[4];		// Max vector size is 4

	this->getData(databuf);
	for (int i = 0; i <vec.size(); i++)
		vec(i) = static_cast<double *>(databuf)[i];

	delete[]databuf;
}
