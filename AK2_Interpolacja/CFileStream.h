#pragma once
#include <iostream>
#include <fstream>
#include <string>

class CFileStream
{
public:
	~CFileStream();
	void openFile();
	double* readData(double* array, int & size);
	void write(double* array, int size);
	

private:
	std::ifstream fileRead; // odczyt z pliku
	std::ofstream fileWrite; // wypis do pliku ( wypis wynikow )
	std::string filename; // pole z nazwa pliku
};
