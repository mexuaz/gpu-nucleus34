#include "defs.hpp"


void _ASERT_(const char* expr_str,
		 bool expr,
		 const char* file,
		 const char* func,
		 size_t line,
		 const std::string& msg)
{
	if (!expr) {
		std::stringstream oss;
		oss << "Assert failed: " << msg << std::endl
		    << "Expected: " << expr_str << std::endl
		    << "Ref >> " << file << '[' << func << "]:" << line << std::endl;
		std::cerr << oss.str();
		throw std::runtime_error(oss.str());
	}
}