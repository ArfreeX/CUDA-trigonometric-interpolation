#include "stdafx.h"

CFileStream::~CFileStream()
{
	fileRead.close();
	fileWrite.close();
}

void CFileStream::openFile()
{
	do
	{
		std::cin >> filename;
		fileRead.open(filename, std::ios::in);

		if (fileRead.good() == true)
			std::cout << "Done, you can now read the data\n";
		else
			std::cout << "File not found, try again or type ""exit"" to leave\n";
	} while (fileRead.good() != true && filename != "exit");
}

double* CFileStream::readData(double* array, int & size)
{
	double val;
	if (fileRead.is_open())
	{
		fileRead >> size;											// read number of rows/columns
		if (fileRead.fail())
			std::cout << "File error - READ SIZE" << std::endl;
		else
			if (size > 0)
			{
				bool argument = true;
				array = new double[size*2];
				for (int i = 0; i < size; )
				{
					fileRead >> val;
					if (fileRead.fail())
					{
						std::cout << "File error - READ DATA" << std::endl;
						break;
					}
					else if (argument)
					{
						array[i] = val;
						argument = false;
					}
					else
					{
						array[size + i] = val;
						++i;
						argument = true;
					}

				}
				return array;
				fileRead.close();
			}
			else
				fileRead.close();
	}
	else
		std::cout << "File error - OPEN" << std::endl;
	return nullptr;
}

void CFileStream::write(int* array, int size)
{
	std::cin >> filename;
	fileWrite.open(filename, std::ios::out | std::ios::app);
	for (int i = 0; i < size; i++)
	{
		fileWrite << array[i] << std::endl;
	}
	fileWrite.close();
}



