#ifndef DEF_VER_HPP_
#define DEF_VER_HPP_

#if defined(__CUDACC__)
#include <thrust/version.h>
#endif


#define STRINGIFY(x) #x
#define VER_STRING(major, minor, patch) STRINGIFY(major) "." STRINGIFY(minor) "-" STRINGIFY(patch)
#define VER_STRING4(major, minor, subminor, patch) STRINGIFY(major) "." STRINGIFY(minor) "." STRINGIFY(subminor) "-" STRINGIFY(patch)

#ifdef __clang__
#define CXX_VER "clang " VER_STRING(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#define CXX_VER VER_STRING(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#endif


namespace { // anonymous

inline std::string tbb_version_string() {
#if defined(TBB_VERSION_MAJOR) && defined(TBB_VERSION_MINOR)
    return std::to_string(TBB_VERSION_MAJOR) + "."
         + std::to_string(TBB_VERSION_MINOR)
  #if defined(TBB_INTERFACE_VERSION)
         + "-" + std::to_string(TBB_INTERFACE_VERSION)
  #endif
         ;
#else
    return "Unknown";
#endif
}

inline std::string thrust_version_string() {
#if defined(THRUST_VERSION)
    // THRUST_VERSION encodes as major * 100000 + minor * 100 + patch
    int major = THRUST_VERSION / 100000;
    int minor = (THRUST_VERSION / 100) % 1000;
    int patch = THRUST_VERSION % 100;
    return std::to_string(major) + "."
         + std::to_string(minor) + "."
         + std::to_string(patch);
#elif defined(THRUST_MAJOR_VERSION) && defined(THRUST_MINOR_VERSION)
    return std::to_string(THRUST_MAJOR_VERSION) + "."
         + std::to_string(THRUST_MINOR_VERSION)
  #if defined(THRUST_SUBMINOR_VERSION)
         + "." + std::to_string(THRUST_SUBMINOR_VERSION)
  #endif
         ;
#else
    return "Unknown";
#endif
}

} // end of anonymous namespace

#endif // DEF_VER_HPP_