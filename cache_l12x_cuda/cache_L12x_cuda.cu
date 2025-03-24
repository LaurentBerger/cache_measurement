#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NB_PAS_MAX 81
#define TPS_MAX_PAR_TEST 2
#define TYPEREEL float





#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
FILETIME a, b, c, d;
// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes
std::pair<std::chrono::steady_clock::time_point, double> getWallTimeAndCpuTime()
{
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		double cpu = (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
		return std::pair<std::chrono::steady_clock::time_point, double>(std::chrono::steady_clock::now(), cpu);
	}
	else {
		//  Handle error
		return std::pair<std::chrono::steady_clock::time_point, double>(std::chrono::steady_clock::now(), -1);
	}
}
// https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g
unsigned long long getTotalSystemMemory()
{
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	return status.ullTotalPhys;
}
#else
#include <unistd.h>

unsigned long long getTotalSystemMemory()
{
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	return pages * page_size;
}
std::pair<std::chrono::steady_clock::time_point, double> getWallTimeAndCpuTime()
{
	return std::pair<std::chrono::steady_clock::time_point, double>(std::chrono::steady_clock::now(), std::clock() / double(CLOCKS_PER_SEC));
}
#endif


__global__ void initTab(TYPEREEL* tab, TYPEREEL val, int nbElt)
{
	for (int idx = 0; idx < nbElt; idx++)
		tab[idx] = val+idx;
}

__global__ void addTestLoop(TYPEREEL* tabA, TYPEREEL* tabB, TYPEREEL* tabC, int nbElt, int nbTest)
{
	for (long long int idxTest = 0; idxTest < nbTest; idxTest++)
		for (int i = 0; i < nbElt; i++)
			tabC[i] = tabA[i] + tabB[i];

}
 

int main() {
	int deviceId;
	int numberOfSMs;
	int deviceCount = 0;
	cudaError_t erreur;

	size_t free, total;
	CUdevice dev;
	CUcontext ctx;
	cuInit(0);
	cuDeviceGet(&dev, 0);
	cuCtxCreate(&ctx, 0, dev);
	CUresult cuRes = cuMemGetInfo(&free, &total);
	if (cuRes != 0)
		std::cout << "Error: " << cudaGetErrorString(cudaError_t(cuRes)) << "\n";
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	std::cout << "free memory : " << free << "\n";
	std::cout << "total memory : " << total << "\n";

	cudaGetDeviceCount(&deviceCount);

	cudaGetDevice(&deviceId);
	cudaSetDevice(deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);



	int nbPas = int(std::log(std::min(int(free / sizeof(TYPEREEL) / 3 /4), int(pow(2.0, NB_PAS_MAX / 3.0)))) / std::log(2))*3;
	if (nbPas <= 0)
	{
		std::cout << "Something went wrong...";
		return -1;
	}
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = 1;
	int nbEltMax = int(pow(2.0, nbPas / 3.0));
	TYPEREEL*tabA, *tabB, *tabC;
	cudaMallocManaged(&tabA, sizeof(TYPEREEL) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	cudaMallocManaged(&tabB, sizeof(TYPEREEL) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	cudaMallocManaged(&tabC, sizeof(TYPEREEL) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	double tps = 0;
	while (tps < TPS_MAX_PAR_TEST)
	{
		nbTest *= 2;
		int nbElt = int(pow(2.0, 7 / 3.0));
		initTab << <1, 1 >> > (tabA, 2.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab << <1, 1 >> > (tabB, 3.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab << <1, 1 >> > (tabC, 0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		std::pair<std::chrono::steady_clock::time_point, double> tpsDeb = getWallTimeAndCpuTime(), tpsFin;
		addTestLoop << <1, 1 >> > (tabA, tabB, tabC, nbElt, nbTest);
		cudaDeviceSynchronize();
		tpsFin = getWallTimeAndCpuTime();
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		tps = tpsFin.second - tpsDeb.second;

	}
	std::cout << "Nombre de tests : " << nbTest << " pour un temps de " << tps << "\n";
	for (int idx = 7; idx < nbPas;idx++)
	{
		
		int nbElt = int(pow(2.0, idx / 3.0));
		initTab <<<1, 1 >>> (tabA, 2.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab <<<1, 1 >>> (tabB, 3.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab <<<1, 1 >>> (tabC, 0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		if (tpsPre > TPS_MAX_PAR_TEST)
			nbTest /= 2;
		if (nbTest == 0)
			nbTest = 1;
		std::pair<std::chrono::steady_clock::time_point, double> tpsDeb = getWallTimeAndCpuTime(), tpsFin;
		addTestLoop <<<1, 1 >>> (tabA, tabB, tabC, nbElt, nbTest);
		cudaDeviceSynchronize();
		tpsFin = getWallTimeAndCpuTime();
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		double tpsCpu = tpsFin.second - tpsDeb.second;
		double tpsWall = double(std::chrono::time_point_cast<std::chrono::milliseconds>(tpsFin.first).time_since_epoch().count());
		tpsWall -= double(std::chrono::time_point_cast<std::chrono::milliseconds>(tpsDeb.first).time_since_epoch().count());
		tpsWall /= 1000;
		std::cout << "<-- " << nbElt << ", " << nbTest << " -->\n";
		std::cout << "Duree : " << tpsWall << " s " << tpsWall / nbTest << "s (" << tpsWall / nbTest / nbElt << "s par element)\t";
		std::cout << "Duree cpu : " << tpsCpu << " s " << tpsCpu / nbTest << "s (" << tpsCpu / nbTest / nbElt << "cpu par element) nbTest=" << nbTest << "\n";
		rapport << tpsWall / nbTest / nbElt << "\t" << tpsWall << "\t";
		rapport << tpsCpu / nbTest / nbElt << "\t" << tpsCpu;
		rapport.flush();
		rapport << "\n";
		tpsPre = tpsWall;

	}
	cudaFree(tabA);
	cudaFree(tabB);
	cudaFree(tabC);
	rapport.close();
	std::cout << "\nEND\n";
    return 0;
}

