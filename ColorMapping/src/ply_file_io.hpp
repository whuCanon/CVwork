
#include <Eigen/Dense>

#include <pcl/io/ply_io.h>

#include <iostream>
#include <vector>
#include <string>


using namespace std;
using namespace pcl;
using namespace Eigen;

typedef unsigned char uchar;
typedef Matrix<uchar, 3, -1> MatrixRGB;


class MyPolygonMesh : public pcl::PolygonMesh
{
public:
	MyPolygonMesh(string ply_filename);
	MyPolygonMesh() {}
	~MyPolygonMesh() {}

	vector<Eigen::Vector3f> vertexs;
	MatrixRGB colors;

	int vertexNum;

private:

};


MyPolygonMesh::MyPolygonMesh(string ply_filename)
{
	pcl::PLYReader plyReader = pcl::PLYReader();
	plyReader.read(ply_filename, *this);

	// get all vertex from meshData
	this->vertexNum = this->cloud.height * this->cloud.width;
	float *tmpPoint;
	uint8_t tmpPointBuf[12];
	for (int i = 0; i < this->vertexNum; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			if (this->cloud.is_bigendian)
				tmpPointBuf[j] = this->cloud.data[i * this->cloud.point_step + (3 - (j % 4) + j / 4 * 4)];
			else
				tmpPointBuf[j] = this->cloud.data[i * this->cloud.point_step + j];
		}
		tmpPoint = (float*) tmpPointBuf;
		this->vertexs.push_back(Vector3f(tmpPoint));
	}
}
