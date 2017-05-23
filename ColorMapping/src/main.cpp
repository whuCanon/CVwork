#include <omp.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef DEBUG
#include <ctime>
#include <fstream>
#endif // DEBUG

#include <iostream>
#include <vector>
#include <string>

#include "ply_file_io.hpp"
#include "image_handler.hpp"

#define IMAGE_ROW 2848
#define IMAGE_COL 4272

using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;


void getMapMatrix(MyPolygonMesh &mesh, MatrixXi &mat, MatrixXi &pixels, VectorXd depth, Matrix<vector<int>, -1, -1> &pixel_isOccupyed, int index);

/*  shading the final color assignment for each vetex by simply compute the
average of the corresponding colors in images.*/
void shading(MyPolygonMesh &mesh, vector<string> &images, vector<MatrixXd> &T, MatrixXi &mapMat);

inline void rendering(MatrixXi & pixels, Matrix<vector<int>, -1, -1> &pixel_isOccupyed, vector<uint32_t> &v, int index_face);
inline void getExtremun(int &min_x, int &max_x, int value_1, int value_2, int value_3);
inline bool pointInTriangle(Vector2i A, Vector2i B, Vector2i C, Vector2i P);

int main(int argc, char **argv)
{
#ifdef DEBUG
	time_t time_clock = time(0);
	ofstream debug_stream;
	debug_stream.open("D:\\debug.log");
#endif // DEBUG

	string input_file_ply = argv[1];
	string input_dir_images = argv[2];
	string input_dir_h5 = argv[3];
	string input_file_h5 = argv[4];
	string output_file_ply = argv[5];

	MyPolygonMesh myMesh(input_file_ply);
	vector<string> images = getFilesInDir(input_dir_images, ".jpg");
	vector<MatrixXd> T = getKTInFile(input_dir_h5, input_file_h5, true);	//	the Eulerian matrixs of all image
	vector<MatrixXd> KT = getKTInFile(input_dir_h5, input_file_h5);	//	the projection matrix in 3*4 size
	/*  the mapping matrix between image and corresponding visible vertexs.
		If there is a mapping, the value in mapMat(i, j) is the pixel of
		vertexs(j) in image(i), stored in 'int' which x in higher 16 bits
		and y in lower 16 bits, 0 otherwise. */
	MatrixXi mapMat(MatrixXi::Zero(images.size(), myMesh.vertexNum));

	MatrixXd vertexs(4, myMesh.vertexNum);				// the homo-coordinates of all vertex in world space
	MatrixXd points(3, myMesh.vertexNum);				// the homo-coordinates of all vertex in one image plain
	MatrixXi pixel_of_v(2, myMesh.vertexNum);			// the coordinate pixel of all vertex in one image
	/*  the occupancy of vetexs for a particular image.
		If pixel(i,j) is occupyed in the image by a vetex, the value
		is the index of this vetex, 0 otherwise.*/
	Matrix<vector<int>, -1, -1> pixel_isOccupyed(IMAGE_ROW, IMAGE_COL);

	for (int i = 0; i < myMesh.vertexNum; i++)
		vertexs.col(i) << (Vector4d() << myMesh.vertexs.at(i).cast<double>(), 1).finished();
	int image_size = images.size();

#ifdef DEBUG
	debug_stream << "initializing time:\t" << time(0) - time_clock << endl;
	time_clock = time(0);
#endif // DEBUG

#ifdef _USE_OMP
	int thread_id;
	int processorNum = 7;

	MatrixXd *OMP_points = new MatrixXd [processorNum];
	MatrixXi *OMP_pixel_of_v = new MatrixXi[processorNum];
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_0(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_1(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_2(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_3(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_4(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_5(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> pixel_isOccupyed_6(IMAGE_ROW, IMAGE_COL);
	Matrix<vector<int>, -1, -1> *OMP_pixel_isOccupyed = new Matrix<vector<int>, -1, -1>[7];
	OMP_pixel_isOccupyed[0] = pixel_isOccupyed_0;
	OMP_pixel_isOccupyed[1] = pixel_isOccupyed_1;
	OMP_pixel_isOccupyed[2] = pixel_isOccupyed_2;
	OMP_pixel_isOccupyed[3] = pixel_isOccupyed_3;
	OMP_pixel_isOccupyed[4] = pixel_isOccupyed_4;
	OMP_pixel_isOccupyed[5] = pixel_isOccupyed_5;
	OMP_pixel_isOccupyed[6] = pixel_isOccupyed_6;

	for (int i = 0; i < processorNum; i++)
	{
		OMP_points[i] = MatrixXd(3, myMesh.vertexNum);
		OMP_pixel_of_v[i] = MatrixXi(2, myMesh.vertexNum);
		//OMP_pixel_isOccupyed[i] = pixel_isOccupyed.replicate(1,1);
	}

	omp_set_num_threads(processorNum);
	
	#pragma omp parallel for \
	    shared(KT, mapMat, vertexs, OMP_points, OMP_pixel_of_v, OMP_pixel_isOccupyed) \
	    private(thread_id)
	for (int i = 0; i < image_size; i++)
	{
		thread_id = omp_get_thread_num();

		OMP_points[thread_id] << KT.at(i) * vertexs;
		OMP_points[thread_id].row(0) << OMP_points[thread_id].row(0).cwiseQuotient(OMP_points[thread_id].row(2));
		OMP_points[thread_id].row(1) << OMP_points[thread_id].row(1).cwiseQuotient(OMP_points[thread_id].row(2));
		OMP_pixel_of_v[thread_id] << OMP_points[thread_id].topRows(2).cast<int>();

		getMapMatrix(myMesh, mapMat, OMP_pixel_of_v[thread_id], OMP_points[thread_id].row(2), OMP_pixel_isOccupyed[thread_id], i);
		cout << "step\t" << i << endl;
	}

	delete[] OMP_points;
	delete[] OMP_pixel_of_v;
	//delete[] OMP_pixel_isOccupyed;
#else
	for (int i = 0; i < image_size; i++)
	{
		points << KT.at(i) * vertexs;
		points.row(0) << points.row(0).cwiseQuotient(points.row(2));
		points.row(1) << points.row(1).cwiseQuotient(points.row(2));
		pixel_of_v << points.topRows(2).cast<int>();
		
		getMapMatrix(myMesh, mapMat, pixel_of_v, points.row(2), pixel_isOccupyed, i);
		cout << "step\t" << i << endl;
	}
#endif // _USE_OMP

#ifdef DEBUG
	debug_stream << "compute mapping time:\t" << time(0) - time_clock << endl;
	time_clock = time(0);
#endif // DEBUG

	shading(myMesh, images, T, mapMat);

#ifdef DEBUG
	debug_stream << "shading time:\t" << time(0) - time_clock << endl;
	time_clock = time(0);
#endif // DEBUG

	myMesh.writeMesh(output_file_ply);

#ifdef DEBUG
	debug_stream << "writing mesh time:\t" << time(0) - time_clock << endl;
	debug_stream.close();
#endif // DEBUG

	return 0;
}


void getMapMatrix(MyPolygonMesh &mesh, MatrixXi &mat, MatrixXi &pixels, VectorXd depth, Matrix<vector<int>, -1, -1> &pixel_isOccupyed, int index)
{
	for (int i = 0; i < mesh.polygons.size(); i++)
		rendering(pixels, pixel_isOccupyed, mesh.polygons.at(i).vertices, i);

	double min_depth = 0;
	int min_index = 0;
	for (int i = 0; i < IMAGE_ROW; i++)
	{
		for (int j = 0; j < IMAGE_COL; j++)
		{
			if (pixel_isOccupyed(i, j).empty())
				continue;

			min_index = pixel_isOccupyed(i, j)[0];
			min_depth = depth(mesh.polygons.at(min_index).vertices[0]);
			for each (int k in pixel_isOccupyed(i,j))
			{
				if (depth(mesh.polygons.at(k).vertices[0]) < min_depth)
				{
					min_index = k;
					min_depth = depth(mesh.polygons.at(k).vertices[0]);
				}
			}
			mat(index, mesh.polygons.at(min_index).vertices[0]) = (pixels(0, mesh.polygons.at(min_index).vertices[0]) << 16) + pixels(1, mesh.polygons.at(min_index).vertices[0]);
			mat(index, mesh.polygons.at(min_index).vertices[1]) = (pixels(0, mesh.polygons.at(min_index).vertices[1]) << 16) + pixels(1, mesh.polygons.at(min_index).vertices[1]);
			mat(index, mesh.polygons.at(min_index).vertices[2]) = (pixels(0, mesh.polygons.at(min_index).vertices[2]) << 16) + pixels(1, mesh.polygons.at(min_index).vertices[2]);

			pixel_isOccupyed(i, j).swap(vector<int>());
		}
	}
}


void shading(MyPolygonMesh &mesh, vector<string> &images, vector<MatrixXd> &T, MatrixXi &mapMat)
{
	//Mat_<cv::Vec3b> temp_image;
	Mat temp_image;

	int vertexNum = mesh.vertexNum;
	int imageNum = images.size();

	float weight;
	uint16_t pixel[2];
	Vector3d vertexNormal;
	MatrixXf sumOfWeight(MatrixXf::Zero(1, vertexNum));

	#ifdef _USE_OMP
		#pragma omp parallel for \
		    shared(mesh, images, T, mapMat, sumOfWeight, imageNum, vertexNum) \
		    private(temp_image, pixel, vertexNormal, weight)
	#endif // _USE_OMP

	for (int i = 0; i < imageNum; i++)
	{
		temp_image = imread(images.at(i)); 
		for (int j = 0; j < vertexNum; j++)
		{
			if (mapMat(i, j))
			{
				*(uint32_t*)pixel = mapMat(i, j);
				vertexNormal = (T.at(i) * (Vector4d() << mesh.normals.col(j).cast<double>(), 1).finished()).head(3);
				weight = pow(vertexNormal.normalized().dot(Vector3d(0, 0, 1)), 2);

				mesh.colors.col(j) += weight * (*(temp_image.ptr<Matrix<uchar, 3, 1>>(pixel[0], pixel[1]))).cast<float>();
				//pixel[0] = mapMat(i, j) >> 16;							//	pixel_x
				//pixel[1] = mapMat(i, j) & 0x0000FFFF;						//	pixel_y
				//mesh.colors(0, j) += temp_image(pixel[1], pixel[0])[0];	//	in opencv, col_0  col_1  col_2  ...
				//mesh.colors(1, j) += temp_image(pixel[1], pixel[0])[1];	//	     row_0 b,g,r  b,g,r  b,g,r  ...
				//mesh.colors(2, j) += temp_image(pixel[1], pixel[0])[2];	//	     row_1 b,g,r  b,g,r  b,g,r  ...
				
				sumOfWeight(0, j) += weight;
			}
		}
	}

	mesh.colors.row(0) << mesh.colors.row(0).cwiseQuotient(sumOfWeight);
	mesh.colors.row(1) << mesh.colors.row(1).cwiseQuotient(sumOfWeight);
	mesh.colors.row(2) << mesh.colors.row(2).cwiseQuotient(sumOfWeight);
}


inline void rendering(MatrixXi & pixels, Matrix<vector<int>, -1, -1> &pixel_isOccupyed, vector<uint32_t> &v, int index_face)
{
	int min_x, min_y, max_x, max_y;

	getExtremun(min_x, max_x, pixels(0, v[0]), pixels(0, v[1]), pixels(0, v[2]));
	getExtremun(min_y, max_y, pixels(1, v[0]), pixels(1, v[1]), pixels(1, v[2]));

	for (int x = min_x; x < max_x; x++)
		for (int y = min_y; y < max_y; y++)
			if (pointInTriangle(pixels.col(v[0]), pixels.col(v[1]), pixels.col(v[2]), Vector2i(x, y)))
				pixel_isOccupyed(x, y).push_back(index_face);
}

inline void getExtremun(int &min_x, int &max_x, int value_1, int value_2, int value_3)
{
	min_x = value_1;
	max_x = min_x;
	if (min_x > value_2)
		min_x = value_2;
	else
		max_x = value_2;
	if (min_x > value_3)
		min_x = value_3;
	else if (max_x < value_3)
		max_x = value_3;
}

inline bool pointInTriangle(Vector2i A, Vector2i B, Vector2i C, Vector2i P)
{
	Vector2i v0 = C - A;
	Vector2i v1 = B - A;
	Vector2i v2 = P - A;

	int dot00 = v0.dot(v0);
	int dot01 = v0.dot(v1);
	int dot02 = v0.dot(v2);
	int dot11 = v1.dot(v1);
	int dot12 = v1.dot(v2);

	float inverDeno = 1.0 / (dot00 * dot11 - dot01 * dot01);

	float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
	if (u < 0 || u > 1) // if u out of range, return directly
	{
		return false;
	}

	float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
	if (v < 0 || v > 1) // if v out of range, return directly
	{
		return false;
	}

	return u + v <= 1;
}
