import numpy as np
import os
from matplotlib import pyplot as plt
import csv

#nom coreP coreE coreLPE thread L1 L2 L3
liste_processeur = {
# cpuz_x64
'i9-13900KF_3.00GHz': (8, 16, 0, 32, 49152, 2097152, 37748736),
# https://valid.x86.fr/ktz0ql
'Ultra7_155Hx22_AC': (6, 8, 2, 22, 49152, 2097152, 25165824),
}

def lire_rapport_csv(nom_fichier):
    nbLigne = 0
    nbColonne=0
    tab_data = []
    label = []
    """
    with open(nom_fichier, 'r') as csvfile:
        content = csvfile.read()
    content =  content.replace('\t\n', '\n')        
    with open(nom_fichier, 'w') as csvfile:
        csvfile.write(content)
    """
    with open(nom_fichier, 'r') as csvfile:
        content = csv.reader(csvfile, delimiter='\t')
        for ligne in content:
            if nbColonne==0:
                nbColonne =  len(ligne)
            if nbColonne != len(ligne):
                print("ERREUR ", nom_fichier, nbLigne, nbColonne, len(ligne)    )
            else:
                d = [float(v) for v in ligne]
                tab_data.append(d)
                label.append(d[0])
                nbLigne = nbLigne + 1 
    return tab_data


if os.name == 'nt':
    dossier_rapport = './'
else:
    dossier_rapport = './'
liste_dossier=[
               './tps_fct_mem_',  './tps_fct_mem_', './tps_fct_mem_',
               './tps_fct_mem_', './tps_fct_mem_'] 
col_use = []
idx = 0
for nom_processeur, nom_dossier in zip(liste_processeur.keys(), liste_dossier):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    tab_data = lire_rapport_csv(nom_dossier + nom_processeur + '.txt')
    print(liste_processeur[nom_processeur][4:])
    x = np.array(tab_data)
    legende = []
    courbe = ax.semilogx(x[:,0]*8*3, x[:,2], color='b', marker='+', base=2)
    col_use.append(courbe[0].get_color())
    ax.set_xlabel('Memory size (Byte)')
    ax.set_ylabel('Time/per double (s)')
    ax.grid(True)
    taille_cache = liste_processeur[nom_processeur][4:7]
    ax.vlines(taille_cache[0], 0, np.max(x[:,2]), colors='g')
    ax.vlines(taille_cache[1], 0, np.max(x[:,2]), colors='r')
    ax.vlines(taille_cache[2], 0, np.max(x[:,2]), colors='k')
    ax.legend([nom_processeur, 'L1 cache size', 'L2 cache size', 'L3 cache size'])
    idx = idx + 1
plt.show()    
