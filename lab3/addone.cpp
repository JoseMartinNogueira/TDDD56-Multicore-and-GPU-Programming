/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu2.hpp>

float addOneFunc(float a)
{
	return a+1;
}


int main(int argc, const char* argv[])
{
	/* Program parameters */
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	
	/* Skeleton instances */
	auto addOneMap = skepu2::Map<1>(addOneFunc);
	
	/* SkePU containers */
	skepu2::Vector<float> input(size), res(size);
	input.randomize(0, 9);
	
	
	// This is how to measure execution times with SkePU
	auto dur = skepu2::benchmark::measureExecTime([&]
	{
		// Code to be measured here
		addOneMap(res, input);
	});
	
	/* This is how to print the time */
	std::cout << "Time: " << (dur.count() / 10E6) << " seconds.\n";
	
	
	/* Print vector for debugging */
	std::cout << "Input:  " << input << "\n";
	std::cout << "Result: " << res << "\n";
	
	
	return 0;
}

