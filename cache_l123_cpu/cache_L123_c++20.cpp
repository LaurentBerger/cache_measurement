﻿#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include  <cfloat>

#define NB_PAS 81
#define TPS_MAX_PAR_TEST 10
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

// Ultra 7 155H https://www.eatyourbytes.com/fr/cpu-detail/intel-ultra-7-155h/
// https://www.eatyourbytes.com/fr/cpu-detail/intel-i9-13900k/
// https://www.eatyourbytes.com/fr/cpu-detail/intel-i7-8650u/
// https://www.eatyourbytes.com/fr/cpu-detail/intel-i5-8400/
// https://www.techpowerup.com/cpu-specs/core-i7-5820k.c1763

void initTab(TYPEREEL* tab, TYPEREEL val, int nbElt)
{
	for (int idx = 0; idx < nbElt; idx++)
		tab[idx] = val+idx;
}


int main() {
    int maxMem = getTotalSystemMemory() / 4;
	bool init = true;
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = 1;
	int nbEltMax = int(getTotalSystemMemory() / 32 / sizeof(TYPEREEL));
	TYPEREEL*tabA = new TYPEREEL[nbEltMax], *tabB = new TYPEREEL[nbEltMax], *tabC = new TYPEREEL[nbEltMax];


	double tps = 0;
	while (tps < TPS_MAX_PAR_TEST)
	{
		nbTest *= 2;
		int nbElt = int(pow(2.0, 7 / 3.0));
		std::pair<std::chrono::steady_clock::time_point, double> tpsDeb = getWallTimeAndCpuTime(), tpsFin;
		for (long long int idxTest = 0; idxTest < nbTest; idxTest++)
			for (int i = 0; i < nbElt; i++)
				tabC[i] = tabA[i] + tabB[i];
		tpsFin = getWallTimeAndCpuTime();
		tps =  tpsFin.second - tpsDeb.second;

	}
	std::cout << "Nombre de tests : " << nbTest << " pour un temps de " << tps << "\n";



	for (int idx = 7; idx < NB_PAS;idx++)
	{
		int nbElt = int(pow(2.0, idx / 3.0));
		initTab(tabA, 2.0, nbElt);
		initTab(tabB, 3.0, nbElt);
		initTab(tabC, 0, nbElt);
		double tpsParTest = 0;
		if (tpsPre > TPS_MAX_PAR_TEST)
			nbTest /= 2;
		if (nbTest<1)
			nbTest = 1;
		rapport << nbElt << "\t" << nbTest << "\t";
		int idxPre = 0;
		std::pair<std::chrono::steady_clock::time_point, double> tpsDeb = getWallTimeAndCpuTime(), tpsFin;
		double tpsMin = DBL_MAX;
		for (long long int idxTest = 0; idxTest < nbTest; idxTest++)
		{
			for (int i = 0; i < nbElt; i++)
				tabC[i] = tabA[i] + tabB[i];
		}
		tpsFin = getWallTimeAndCpuTime();
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
	delete tabA;
	delete tabB;
	delete tabC;
	rapport.close();
    return 0;
}

