
#include <boost/filesystem.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <string>

#include "hdf5_file_reader.hpp"


using namespace std;
using namespace Eigen;

vector<string> getFilesInDir(string directory, string img_suffix);
vector<MatrixXd> getTInFile(string T1_dir, string T2_file);


vector<string> getFilesInDir(string directory, string format)
{
	namespace fs = boost::filesystem;
	fs::path dir(directory);

	std::cout << "path: " << directory << "    Loading......    ";
	if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
		PCL_THROW_EXCEPTION(pcl::IOException, "No valid directory given!\n");

	vector<string> result;
	fs::directory_iterator pos(dir);
	fs::directory_iterator end;

	for (; pos != end; ++pos)
		if (fs::is_regular_file(pos->status()))
			if (fs::extension(*pos) == format)
				result.push_back(pos->path().string());

	cout << "Done." << endl;
	sort(result.begin(), result.end());
	return result;
}

vector<MatrixXd> getTInFile(string T1_dir, string T2_file)
{
	vector<string> T1_files = getFilesInDir(T1_dir, ".h5");
	
	vector<MatrixXd> result;
	HDF5FileReader fileReader;
	MatrixXd T1(4, 4);
	MatrixXd T2(3, 4);
	MatrixXd K(3, 3);
	string tmp_dsetname;
	
	for (int i = 0; i < 5; i++)
	{
		tmp_dsetname = "H_N" + to_string(i + 1) + "_from_NP5";
		fileReader.setFilename(T2_file);
		fileReader.setDatasetname(tmp_dsetname);
		fileReader.getMatrix(T2);

		tmp_dsetname = "N" + to_string(i + 1) + "_rgb_K";
		fileReader.setFilename(T2_file);
		fileReader.setDatasetname(tmp_dsetname);
		fileReader.getMatrix(K);

		for (int j = 0; j < T1_files.size(); j++)
		{
			tmp_dsetname = "H_table_from_reference_camera";
			fileReader.setFilename(T1_files.at(j));
			fileReader.setDatasetname(tmp_dsetname);
			fileReader.getMatrix(T1);

			result.push_back(K * T2 * T1.inverse());
		}
	}
	return result;
}
