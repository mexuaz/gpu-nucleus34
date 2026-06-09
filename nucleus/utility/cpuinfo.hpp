#ifndef __CPUINFO__H
#define __CPUINFO__H

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#endif


#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#define _NVDA_DEVINF_NAME "Name"
#define _NVDA_DEVINF_CM_CAP "Compute capability"
#define _NVDA_DEVINF_CLK_RATE "Clock rate"
#define _NVDA_DEVINF_CPY_OVERLAP "Device copy overlap"
#define _NVDA_DEVINF_EXE_TIMEOUT "Kernel execution timeout"
#define _NVDA_DEVINF_MEM_GLOBAL "Total global mem"
#define _NVDA_DEVINF_MEM_CONST "Total constant Mem"
#define _NVDA_DEVINF_MEM_PITCH "Max mem pitch"
#define _NVDA_DEVINF_TEX_ALIGN "Texture Alignment"
#define _NVDA_DEVINF_MP_COUNT "Multiprocessor count"
#define _NVDA_DEVINF_MEM_PER_BLK "Shared mem per mp"
#define _NVDA_DEVINF_REG_PER_BLK "Registers per mp"
#define _NVDA_DEVINF_THREADS_WRAP "Threads in warp"
#define _NVDA_DEVINF_MAX_THREADS_BLK "Max threads per block"
#define _NVDA_DEVINF_MAX_THREAD_DIM "Max thread dimensions"
#define _NVDA_DEVINF_MAX_GRID_DIM "Max grid dimensions"

using puinfo_t = std::map<std::string, std::string>; // CPU/GPU info type
using procs_t = std::vector<puinfo_t>; // CPUs/GPUs type

procs_t gpuinfo( void ) {
	procs_t procs;
#if defined(__CUDACC__)
	int count;
	cudaGetDeviceCount( &count );
	procs.resize(count);
	for (int i = 0; i < count; i++) {
		cudaDeviceProp  prop;
		cudaGetDeviceProperties( &prop, i );

		auto& inf = procs[i];
		inf[_NVDA_DEVINF_NAME] = prop.name;
		inf[_NVDA_DEVINF_CM_CAP] =             std::to_string(prop.major)
							+ "." +
							std::to_string(prop.minor);
		{
			int clockRateKHz = 0;
			cuDeviceGetAttribute(&clockRateKHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, i);
			inf[_NVDA_DEVINF_CLK_RATE] = std::to_string(clockRateKHz);
		}
		inf[_NVDA_DEVINF_CPY_OVERLAP] =            prop.asyncEngineCount > 0 ? "Enabled" : "Disabled";
		inf[_NVDA_DEVINF_EXE_TIMEOUT] =       "N/A";
		inf[_NVDA_DEVINF_MEM_GLOBAL] =               std::to_string(prop.totalGlobalMem);
		inf[_NVDA_DEVINF_MEM_CONST] =             std::to_string(prop.totalConstMem);
		inf[_NVDA_DEVINF_MEM_PITCH] =                  std::to_string(prop.memPitch);
		inf[_NVDA_DEVINF_TEX_ALIGN] =              std::to_string(prop.textureAlignment);
		inf[_NVDA_DEVINF_MP_COUNT] =           std::to_string(prop.multiProcessorCount);
		inf[_NVDA_DEVINF_MEM_PER_BLK] =              std::to_string(prop.sharedMemPerBlock);
		inf[_NVDA_DEVINF_REG_PER_BLK] =               std::to_string(prop.regsPerBlock);
		inf[_NVDA_DEVINF_THREADS_WRAP] =                std::to_string(prop.warpSize);
		inf[_NVDA_DEVINF_MAX_THREADS_BLK] =          std::to_string(prop.maxThreadsPerBlock );
		inf[_NVDA_DEVINF_MAX_THREAD_DIM] =          std::to_string(prop.maxThreadsDim[0])
				+ "x" + std::to_string(prop.maxThreadsDim[1])
				+ "x" + std::to_string(prop.maxThreadsDim[2]);
		inf[_NVDA_DEVINF_MAX_GRID_DIM] = std::to_string(prop.maxGridSize[0])
				+ "x" + std::to_string(prop.maxGridSize[1])
				+ "x" + std::to_string(prop.maxGridSize[2]);
	} // loop devices
#endif
	return procs;
}


inline procs_t cpuinfo( void ) {

	auto trim = [](const std::string& sz) {
		auto str(sz);
		str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) {return !isspace(ch);})); // left trim
		str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch) {return !isspace(ch);}).base(), str.end()); // right trim
		return str;
	};

	procs_t cpus;
	puinfo_t inf;

	std::ifstream f("/proc/cpuinfo");
	if(!f.is_open()) {
		return cpus;
	}

	std::string strLine;
	while(!f.eof()) {
		std::getline(f, strLine);
		auto pos = strLine.find(':');
		if(pos == std::string::npos && inf.size()) { // (blank line) go to next cpu info
			cpus.push_back(inf);
			inf.clear();
			continue;
		}

		inf[trim(strLine.substr(0, pos))]=trim(strLine.substr(pos+1));

	} // end of while loop eof

	f.close();

	return cpus;
}


inline std::string cpu_affinity() {
	std::string affinity("");
#if defined(__linux__)
	cpu_set_t cpu_set;
	if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set)) {
		throw std::runtime_error("Failed to get cpu affinity.");
	}
	int cpu_first = -1;
	int cpu_last = -1;
	auto make_affinity_string = [&]() {
		if(cpu_first > -1 && cpu_last > -1) {
			affinity += (affinity.empty() ? "" : "-")
					+ std::to_string(cpu_first)
					+ ":" + std::to_string(cpu_last);
			cpu_first = -1;
			cpu_last = -1;
		} else if (cpu_first > -1) {
			affinity += (affinity.empty() ? "" : "-")
					+ std::to_string(cpu_first);
			cpu_first = -1;
		}
	};
	for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
		if (CPU_ISSET(cpu, &cpu_set)) {
			if(cpu_first < 0) {
				cpu_first = cpu;
				continue;
			}
			cpu_last = cpu;
		} else {
			make_affinity_string();
		}
	} // for all CPUs
	make_affinity_string();
#endif
	return affinity;

}

#endif // __CPUINOF__H
