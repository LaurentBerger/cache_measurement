﻿#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include  <cfloat>

#define NB_ELT 2
#define NB_TEST 10000000ll
#define NB_PAS 81
#define TPS_MAX_PAR_TEST 10
#define MIN_CLCK 1


#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
FILETIME a, b, c, d;
inline double getCpuTime()
{
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0)
    {
		return
			(double)(d.dwLowDateTime |
				((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	return 0;
}
#else
inline double getCpuTime()
{
	return std::clock() / double(CLOCKS_PER_SEC);
}
#endif


void initTab(double* tab, double val, int nbElt)
{
	for (int idx = 0; idx < nbElt; idx++)
		tab[idx] = val+idx;
}


int main() {
    int maxMem = NB_ELT *1024 * 256;
	bool init = true;
	double pasMem = (std::log(maxMem) - std::log(NB_ELT))/ NB_PAS;
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = NB_TEST;
	int nbEltMax = NB_ELT * int(pow(2.0, NB_PAS / 3.0));
	double *tabA = new double[nbEltMax], *tabB = new double[nbEltMax], *tabC = new double[nbEltMax];
	for (int idx = 7; idx < NB_PAS;idx++)
	{
		
		int nbElt = NB_ELT * int(pow(2.0, idx / 3.0));
		initTab(tabA, 2.0, nbElt);
		initTab(tabB, 3.0, nbElt);
		initTab(tabC, 0, nbElt);
		double tpsParTest = 0;
		if (tpsPre > TPS_MAX_PAR_TEST)
			nbTest /= 2;
		rapport << nbElt << "\t" << nbTest << "\t";
		double finPre, minClck= 1;
		int idxPre = 0;
		double debut = getCpuTime();
		double tpsMin = DBL_MAX;
		for (long long int idxTest = 0; idxTest < nbTest; idxTest++)
		{
			for (int i = 0; i < nbElt; i++)
				tabC[i] = tabA[i] + tabB[i];
		}
		finPre = getCpuTime();
		double tps = finPre - debut;
		tpsParTest = tps;
		std::cout << "<-- " << nbElt << " -->\nDurée sans thread (" << tpsParTest << " ticks) ";
		tpsPre = tps;
		tpsParTest = tpsParTest  / nbTest;
		std::cout << tpsParTest << "s (" << tpsParTest / nbElt << "s par élément) nbTest=" << nbTest<<"\n";
		rapport << tpsParTest / nbElt << "\t" << tps << "\t" << tpsMin / nbElt ;
		rapport << "\n";
		rapport.flush();

	}
	delete tabA;
	delete tabB;
	delete tabC;
	rapport.close();
    return 0;
}

