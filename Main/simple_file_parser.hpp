#pragma once
#include <vector>
#include <string>
#include <fstream>

namespace orfp 
{
	inline std::vector<float> parse(std::ifstream& fstream)
	{
		if (!fstream.good())
		{
			throw std::exception("Invalid input stream for parsing");
		}

		std::vector<float> values{};
		std::string buffer{};

		while (std::getline(fstream, buffer))
		{
			values.push_back(std::stof(buffer));
		}

		return values;
	}
}
