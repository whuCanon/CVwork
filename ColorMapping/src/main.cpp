
#include <omp.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef DEBUG
#include <fstream>
#endif // DEBUG

#include <iostream>
#include <vector>
#include <string>

#include "ply_file_io.hpp"
#include "image_handler.hpp"

#define IMAGE_ROW 2848
#define IMAGE_COL 4272
#define IMAGE_NUM  600

using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;


void getMapMatrix(MatrixXi &mat, MatrixXi &pixels, VectorXd depth, MatrixXi &pixel_isOccupyed, int index);

/*  shading the final color assignment for each vetex by simply compute the
average of the corresponding colors in images.*/
void shading(MyPolygonMesh &mesh, vector<string> &images, MatrixXi &mapMat);


int main(int argc, char **argv)
{
	string input_file_ply = argv[1];
	string input_dir_images = argv[2];
	string input_dir_h5 = argv[3];
	string input_file_h5 = argv[4];
	string output_file_ply = argv[5];

	MyPolygonMesh myMesh(input_file_ply);
	vector<string> images = getFilesInDir(input_dir_images, ".jpg");
	vector<MatrixXd> T = getTInFile(input_dir_h5, input_file_h5);	//	the projection matrix in 3*4 size
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
	MatrixXi pixel_isOccupyed(MatrixXi::Zero(IMAGE_ROW, IMAGE_COL));

	for (int i = 0; i < myMesh.vertexNum; i++)
		vertexs.col(i) << (Vector4d() << myMesh.vertexs.at(i).cast<double>(), 1).finished();

#ifdef _USE_OMP
	int processorNum = atoi(getenv("NUMBER_OF_PROCESSORS")) - 1;

	vector<MatrixXd> OMP_points;
	vector<MatrixXi> OMP_pixel_of_v;
	vector<MatrixXi> OMP_pixel_isOccupyed;

	for (int i = 0; i < processorNum; i++)
	{
		OMP_points.push_back(MatrixXd(3, myMesh.vertexNum));
		OMP_pixel_of_v.push_back(MatrixXi(2, myMesh.vertexNum));
		OMP_pixel_isOccupyed.push_back(MatrixXi::Zero(IMAGE_ROW, IMAGE_COL));
	}

	int thread_id;
	omp_set_num_threads(processorNum);

	#pragma omp parallel for shared(T, mapMat, vertexs, OMP_points, OMP_pixel_of_v, OMP_pixel_isOccupyed) private(thread_id)
	for (int i = 0; i < IMAGE_NUM; i++)
	{
		thread_id = omp_get_thread_num();

		OMP_points.at(thread_id) << T.at(i) * vertexs;
		OMP_points.at(thread_id).row(0) << OMP_points.at(thread_id).row(0).cwiseQuotient(OMP_points.at(thread_id).row(2));
		OMP_points.at(thread_id).row(1) << OMP_points.at(thread_id).row(1).cwiseQuotient(OMP_points.at(thread_id).row(2));
		OMP_pixel_of_v.at(thread_id) << OMP_points.at(thread_id).topRows(2).cast<int>();

		getMapMatrix(mapMat, OMP_pixel_of_v.at(thread_id), OMP_points.at(thread_id).row(2), OMP_pixel_isOccupyed.at(thread_id), i);
		cout << "step\t" << i << endl;
	}
#else
	for (int i = 0; i < IMAGE_NUM; i++)
	{
		points << T.at(i) * vertexs;
		points.row(0) << points.row(0).cwiseQuotient(points.row(2));
		points.row(1) << points.row(1).cwiseQuotient(points.row(2));
		pixel_of_v << points.topRows(2).cast<int>();
		
		getMapMatrix(mapMat, pixel_of_v, points.row(2), pixel_isOccupyed, i);
		cout << "step\t" << i << endl;
	}
#endif // _USE_OMP

	shading(myMesh, images, mapMat);
	myMesh.writeMesh(output_file_ply);

#ifdef DEBUG
	ofstream debug_stream;
	debug_stream.open("C:\\Temp\\data\\debug.log");

	Mat_<cv::Vec3b> debug_image = imread(images.at(0));
	debug_stream << "r,g,b:\t" << static_cast<int>(debug_image(1100, 2100)[0]) << "\t"
				 << static_cast<int>(debug_image(1100, 2100)[1]) << "\t" 
				 << static_cast<int>(debug_image(1100, 2100)[2]) << endl;

	debug_stream.close();
#endif // DEBUG

	return 0;
}


void getMapMatrix(MatrixXi &mat, MatrixXi &pixels, VectorXd depth, MatrixXi &pixel_isOccupyed, int index)
{
	int vertexNum = depth.size();
	int occupyIndex;

	pixel_isOccupyed.setZero();

	for (int i = 0; i < vertexNum; i++)
	{
		occupyIndex = pixel_isOccupyed(pixels(0, i), pixels(1, i));
		if (!occupyIndex)
		{
			pixel_isOccupyed(pixels(0, i), pixels(1, i)) = i;
			mat(index, i) = (pixels(0, i) << 16) + pixels(1, i);
		}
		else if (depth(i) < depth(occupyIndex))
		{
			mat(index, occupyIndex) = 0;
			pixel_isOccupyed(pixels(0, i), pixels(1, i)) = i;
			mat(index, i) = (pixels(0, i) << 16) + pixels(1, i);
		}
	}
}


void shading(MyPolygonMesh &mesh, vector<string> &images, MatrixXi &mapMat)
{
	Mat_<cv::Vec3b> temp_image;
	int pixel[2];
	int vertexNum = mesh.vertexNum;
	int imageNum = images.size();
	MatrixXi counts(MatrixXi::Zero(1, vertexNum));			//	the count of value

#ifdef _USE_OMP
	#pragma omp parallel for shared(mesh, images, mapMat, counts, imageNum, vertexNum) private(temp_image, pixel)
#endif // _USE_OMP
	for (int i = 0; i < imageNum; i++)
	{
		temp_image = imread(images.at(i));
		for (int j = 0; j < vertexNum && mapMat(i, j); j++)
		{
			pixel[0] = mapMat(i, j) >> 16;
			pixel[1] = mapMat(i, j) & 0x00001111;
			mesh.colors(2, j) += temp_image(pixel[0], pixel[1])[0];
			mesh.colors(1, j) += temp_image(pixel[0], pixel[1])[1];
			mesh.colors(0, j) += temp_image(pixel[0], pixel[1])[2];
			counts(0, j)++;
		}
	}

	mesh.colors.row(0) << mesh.colors.row(0).cwiseQuotient(counts.cast<float>());
	mesh.colors.row(1) << mesh.colors.row(1).cwiseQuotient(counts.cast<float>());
	mesh.colors.row(2) << mesh.colors.row(2).cwiseQuotient(counts.cast<float>());
}
