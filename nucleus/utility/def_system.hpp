#ifndef DEF_SYSTEM_HPP_
#define DEF_SYSTEM_HPP_

#include <string>
#include <cstdlib> // getenv(...)

#if defined(__linux__)

#include <unistd.h> // gethostname(...)
#include <limits.h> // HOST_NAME_MAX

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 64
#endif // HOST_NAME_MAX

#endif // __linux__



inline std::string env(const char* e)
{
    auto v = std::getenv(e);
    if(v)
	{
		return std::string(v);
    }
    return "";
}

inline std::string hostname() {
	std::string str_host;
#if defined(__linux__)
	char hostname[HOST_NAME_MAX];
	auto host = gethostname(hostname, HOST_NAME_MAX);
	if(host) {
		return "";
	}

	str_host.assign(hostname);
#endif
	return str_host;
}

inline auto get_path(const std::string& filename) {
	return filename.substr(0, filename.find_last_of("/\\"));
}

inline auto strip_path(const std::string& filename) {
	return filename.substr(filename.find_last_of("/\\") + 1);
}

inline auto strip_ext(const std::string& filename) {
	return filename.substr(0, filename.find_last_of('.'));
}

inline auto get_ext(const std::string& filename) {
	return filename.substr(filename.find_last_of('.'));
}

#endif // DEF_SYSTEM_HPP_