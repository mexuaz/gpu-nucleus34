#pragma once
#include <sstream>
#include <string>
#include <cstdio>

// Global JSON output stream pointer. Set this to &jsonSS before emitting
// any field()/beginObject() calls; reset to nullptr when done. Header-inline
// so each executable that includes it gets its own definition.
inline std::ostringstream* g_jsonSS = nullptr;

namespace json {

inline std::string escapeString(const std::string& s) {
	std::string out;
	out.reserve(s.size() * 2 + 2);
	out += '"';
	for (unsigned char c : s) {
		switch (c) {
			case '"':  out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\b': out += "\\b";  break;
			case '\f': out += "\\f";  break;
			case '\n': out += "\\n";  break;
			case '\r': out += "\\r";  break;
			case '\t': out += "\\t";  break;
			default:
				if (c < 0x20) {
					char buf[7];
					snprintf(buf, sizeof(buf), "\\u%04x", c);
					out += buf;
				} else {
					out += static_cast<char>(c);
				}
		}
	}
	out += '"';
	return out;
}

inline void field(bool first_field, const std::string& key, const std::string& value) {
	if (!g_jsonSS) return;
	(*g_jsonSS) << (first_field ? "{" : ",") << "\n\t\t\"" << key << "\": " << escapeString(value);
}

inline void field(bool first_field, const std::string& key, const char* value) {
	field(first_field, key, std::string(value ? value : ""));
}

inline void field(bool first_field, const std::string& key, bool value) {
	if (!g_jsonSS) return;
	(*g_jsonSS) << (first_field ? "{" : ",") << "\n\t\t\"" << key << "\": " << (value ? "true" : "false");
}

template <typename T>
inline void field(bool first_field, const std::string& key, const T& value) {
	if (!g_jsonSS) return;
	(*g_jsonSS) << (first_field ? "{" : ",") << "\n\t\t\"" << key << "\": " << value;
}

// Emit a field whose value is a pre-formatted JSON fragment (array, number, etc.)
// without applying string escaping.
inline void raw_field(bool first_field, const std::string& key, const std::string& raw_value) {
	if (!g_jsonSS) return;
	(*g_jsonSS) << (first_field ? "{" : ",") << "\n\t\t\"" << key << "\": " << raw_value;
}

// Emit a JSON array field by streaming each element through `fmt(os, element)`.
template <typename It, typename Fmt>
inline void array_field(bool first_field, const std::string& key, It begin, It end, Fmt fmt) {
	if (!g_jsonSS) return;
	(*g_jsonSS) << (first_field ? "{" : ",") << "\n\t\t\"" << key << "\": [";
	bool first_elem = true;
	for (auto it = begin; it != end; ++it) {
		if (!first_elem) (*g_jsonSS) << ", ";
		fmt(*g_jsonSS, *it);
		first_elem = false;
	}
	(*g_jsonSS) << "]";
}

inline void beginObject(bool first_object, const std::string& key) {
	if (g_jsonSS)
		(*g_jsonSS) << (first_object ? "{" : ",") << "\n\t\"" << key << "\":";
}

inline void endObject() {
	if (g_jsonSS) (*g_jsonSS) << "\n\t}";
}

inline void lastObject() {
	if (g_jsonSS) (*g_jsonSS) << "\n}";
}

} // namespace json
