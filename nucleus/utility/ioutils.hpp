#ifndef _IOUTILS_HPP_
#define _IOUTILS_HPP_

#include <vector>
#include <array>
#include <iterator> // ostream_iterator
#include <istream>
#include <sstream>
#include <iostream>
#include <fstream>



/**
 * We need to warp the following operator in std namespace
 * @cite https://stackoverflow.com/a/24111762/1146097
 */
namespace std {

	template <typename T, size_t SIZE>
        std::ostream& operator<<(std::ostream& out,
			const std::array<T, SIZE>& tp) {
		out << '(';
		for (size_t s = 0; s < SIZE; s++) {
			out << tp[s] << ((s==SIZE-1) ? ')' : ',');
		}
		return out;
	}

	template <typename T>
	std::ostream& operator<< (std::ostream& out,
		const std::vector<T>& v) {
		out << "[ ";
		std::copy (v.cbegin(), v.cend(), std::ostream_iterator<T>(out, " "));
		out << ']';
	return out;
	}
}




inline void file_put_content(const std::string& filename,
                             const std::string& content) {
	std::ofstream ofl(filename);
	std::copy(content.begin(), content.end(),
	          std::ostreambuf_iterator<char>(ofl));
	ofl.close();
}

template <typename Iterator>
inline void file_put_content(const std::string& filename,
			     Iterator begin, Iterator end) {
	std::ofstream ofl(filename);
	for(auto it = begin; it != end; it++) {
		ofl << *it << std::endl;
	}
	ofl.close();
}

inline auto file_get_content(const std::string& filename) {
	std::ifstream ifl(filename);
	// extra parentheses around the first argument is essential
	// most vexing parse
	std::string str((std::istreambuf_iterator<char>(ifl)),
			    std::istreambuf_iterator<char>());
	ifl.close();
	return str;
}

template <typename T>
inline auto file_get_vector(const std::string& filename) {
	std::ifstream ifl(filename);
	std::istream_iterator<T> start(ifl), end;
	return std::vector<T> (start, end);
}

template <typename T>
inline auto file_get_columns(const std::string& filename) {
	std::vector<std::vector<T>> cols;
	std::ifstream ifl(filename);
	for(std::string line; std::getline(ifl, line);) {
		std::vector<T> tokens;
		std::istringstream words(line);
		for(auto it = std::istream_iterator<T>(words); it != std::istream_iterator<T>();it++){
			tokens.push_back(*it);
		} // loop words
		if(tokens.size()) {
			cols.push_back(tokens);
		}
	} // loop lines
	ifl.close();
	return cols;
}

/**
 * @brief explode Split a string based on whitespace
 * 
 * @param str 
 * @return std::vector<std::string> 
 */
inline std::vector<std::string> explode(const std::string& str) {
	std::istringstream iss(str);
	return std::vector<std::string>{
		std::istream_iterator<std::string>{iss},
		std::istream_iterator<std::string>{}};
}

#endif // _IOUTILS_HPP_
