#include "helpers.hpp"


std::string hr_cap(const std::string& val, int precision /*= 2*/) {
	return hr_cap(strtod(val.c_str(), nullptr), precision);
}
