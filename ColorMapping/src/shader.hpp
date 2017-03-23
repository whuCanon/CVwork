
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;
using namespace Eigen;

typedef Matrix<uchar, 3, -1> MatrixRGB;

/*  shading the final color assignment for each vetex by simply compute the 
	average of the corresponding colors in images.*/
MatrixRGB shading(vector<string> &images, MatrixXi &mapMat);


MatrixRGB shading(vector<string>& images, MatrixXi & mapMat)
{
	Mat temp_image;
	int vertexNum = mapMat.row(0).size();
	int imageNum = images.size();
	MatrixXf result(MatrixXf::Zero(4, vertexNum));			//	the return value and count of value

	for (int i = 0; i < imageNum; i++)
	{
		temp_image = imread(images.at(i));
		for (int j = 0; j < vertexNum && mapMat(i, j); j++)
		{
			result.col(j) += *(temp_image.ptr<Vector3f>(mapMat(i, j) >> 16, mapMat(i, j) & 0x00001111));
			result(3, j) += 1;
		}
		cout << "tick" << endl;
	}

	result.row(0) << result.row(0).cwiseQuotient(result.row(3));
	result.row(1) << result.row(1).cwiseQuotient(result.row(3));
	result.row(2) << result.row(2).cwiseQuotient(result.row(3));
	
	return result.topRows(3).cast<uchar>();
}
