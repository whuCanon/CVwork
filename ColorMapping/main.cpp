/* 
 * 
 */


#include <iostream>
#include <pcl/io/ply_io.h>

int main(int argc, char** args)
{
	//  read .ply file to object which represent the mesh, mesh obj = meshRead("*.ply");
	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);
	
	if (pcl::io::loadPLYFile(args[1], *mesh) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read file %s \n", args[1]);
		return (-1);
	}

	//  subdivide the mesh, mesh new_obj = subdivide(mesh);

	//  estimate the transform matrix of each image, input image, output T[n];

	//  get the preliminary result, for each image, act T[i] on the mesh to \
		get the mapping between vertex and pixel, Input mseh, images and T[n], output mesh.

	return (0);
}
