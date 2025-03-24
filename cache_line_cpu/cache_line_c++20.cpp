#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include  <cfloat>

#define NB_TEST 111ll
#define NB_PAS 13
#define TYPEREEL float


#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
FILETIME a, b, c, d;

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


void initTab(TYPEREEL* tab, TYPEREEL val, int nbElt)
{
	for (int idx = 0; idx < nbElt; idx++)
		tab[idx] = val+idx;
}


int main() {
	unsigned long long maxMem = getTotalSystemMemory() / 128;
	int nbEltMax = maxMem / sizeof(TYPEREEL);
	bool init = true;
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = NB_TEST;
	TYPEREEL *tabA = new TYPEREEL[nbEltMax];
	int K = 1;
	for (int idx = 1; idx < NB_PAS;idx++)
	{
		
		int nbElt = nbEltMax;
		initTab(tabA, 2.0, nbElt);
		int idxPre = 0;

		std::pair<std::chrono::steady_clock::time_point, double> tpsDeb = getWallTimeAndCpuTime(), tpsFin;
		for (long long int idxTest = 0; idxTest < nbTest; idxTest++)
		{
			for (int i = 0; i < nbElt; i+=K)
				tabA[i] *= 3;
		}
		tpsFin = getWallTimeAndCpuTime();
		double tpsCpu = tpsFin.second - tpsDeb.second;
		double tpsWall = double(std::chrono::time_point_cast<std::chrono::milliseconds>(tpsFin.first).time_since_epoch().count());
		tpsWall -= double(std::chrono::time_point_cast<std::chrono::milliseconds>(tpsDeb.first).time_since_epoch().count());
		tpsWall /= 1000;
		std::cout << "<-- " << nbElt << ", " << nbTest << ", " << K << ", " << sizeof(TYPEREEL) << " -->\n";
		std::cout << "Duree : " << tpsWall << " s " << tpsWall / nbTest << "s (" << tpsWall / nbTest / nbElt << "s par element)\t";
		std::cout << "Duree cpu : " << tpsCpu << " s " << tpsCpu / nbTest << "s (" << tpsCpu / nbTest / nbElt << "cpu par element) nbTest=" << nbTest << "\n";
		rapport << nbElt << "\t" << nbTest << "\t" << K << "\t" << sizeof(TYPEREEL) << "\t";
		rapport << tpsWall / nbTest / nbElt << "\t" << tpsWall << "\t";
		rapport << tpsCpu / nbTest / nbElt << "\t" << tpsCpu;
		rapport << "\n";
		rapport.flush();
		K *= 2;
	}
	delete tabA;
	rapport.close();
    return 0;
}

