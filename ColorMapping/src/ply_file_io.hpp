
#include <Eigen/Dense>

#include <pcl/io/ply_io.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>


using namespace std;
using namespace pcl;
using namespace Eigen;

typedef unsigned char uchar;


class MyPolygonMesh : public pcl::PolygonMesh
{
public:
	MyPolygonMesh(string ply_filename);
	void writeMesh(string ply_filename);
	MyPolygonMesh() {}
	~MyPolygonMesh() {}

	vector<Eigen::Vector3f> vertexs;
	MatrixXf colors;

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

	this->colors = MatrixXf::Zero(3, this->vertexNum);
}


//void MyPolygonMesh::writeMesh(string ply_filename)
//{
//	PCLPointField colorField = PCLPointField();
//
//	//	write 'r' field
//	colorField.name = "red";
//	colorField.offset = 12;
//	colorField.datatype = 2;
//	colorField.count = 1;
//	this->cloud.fields.push_back(colorField);
//
//	//	write 'g' field
//	colorField.name = "green";
//	colorField.offset = 13;
//	colorField.datatype = 2;
//	colorField.count = 1;
//	this->cloud.fields.push_back(colorField);
//
//	//	write 'b' field
//	colorField.name = "blue";
//	colorField.offset = 14;
//	colorField.datatype = 2;
//	colorField.count = 1;
//	this->cloud.fields.push_back(colorField);
//
//	//	modify 'point_step' and 'row_step'
//	this->cloud.point_step += 3;
//	this->cloud.row_step += 3 * this->cloud.height * this->cloud.width;
//
//	//	write data to 'data'
//	vector<uint8_t> databuf;
//	for (int i = 0; i < this->vertexNum; i++)
//	{
//		for (int j = 0; j < 12; j++)
//			databuf.push_back(this->cloud.data.at(12 * i + j));
//		for (int j = 0; j < 3; j++)
//			databuf.push_back(this->colors(j, i));
//	}
//	this->cloud.data = databuf;
//
//	//	save ply_file
//	pcl::io::savePLYFileBinary(ply_filename, *this);
//}


void MyPolygonMesh::writeMesh(string ply_filename) 
{
	ofstream out_stream;
	out_stream.open(ply_filename);

	//	write header
	out_stream << "ply" << endl;
	out_stream << "format ascii 1.0" << endl;
	out_stream << "comment WentaoLiu generated" << endl;
	out_stream << "element vertex " << this->vertexNum << endl;
	out_stream << "property float x" << endl;
	out_stream << "property float y" << endl;
	out_stream << "property float z" << endl;
	out_stream << "property uchar red" << endl;
	out_stream << "property uchar green" << endl;
	out_stream << "property uchar blue" << endl;
	out_stream << "element face " << this->polygons.size() << endl;
	out_stream << "property list uchar int vertex_indices" << endl;
	out_stream << "end_header" << endl;

	//	write vertexs
	for (int i = 0; i < this->vertexNum; i++)
	{
		out_stream << this->vertexs.at(i)(0) << " ";
		out_stream << this->vertexs.at(i)(1) << " ";
		out_stream << this->vertexs.at(i)(2) << " ";
		out_stream << static_cast<int>(this->colors(0, i)) << " ";
		out_stream << static_cast<int>(this->colors(1, i)) << " ";
		out_stream << static_cast<int>(this->colors(2, i)) << " " << endl;
	}

	//	write face
	for (int i = 0; i < this->polygons.size(); i++)
	{
		out_stream << this->polygons.at(i).vertices[0] << " ";
		out_stream << this->polygons.at(i).vertices[1] << " ";
		out_stream << this->polygons.at(i).vertices[2] << endl;
	}

	out_stream.close();
}
