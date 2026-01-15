# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 19:20:12 2025

@author: Joan
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset
import statsmodels.api as sm
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2_contingency
from scipy.stats import kendalltau
import dcor
from sigfig import round
import seaborn


import sys
sys.path.append(r"C:\Users\Joan\Documents\Física\9e semestre\TFG\grafics")  
from funcions import arrays, taus_1, moments, mitjana_condi

#pugem dades i fem datasets
arxiu = pd.ExcelFile(r"C:\Users\Joan\Documents\Física\9e semestre\TFG\dades separades.xlsx")
df1=arxiu.parse("caleidoscopi")
df2=arxiu.parse("biennal")

df=pd.concat([df1, df2], ignore_index=True)

# Definim una funció per no repetir codi
def prepara_story(df, story, col_dec, col_game, col_time):
    d = df[df[col_game].notna()].copy()
    d = d.assign(story=story)
    d = d[['gender', 'age', 'studies', col_dec, 'story', col_game, col_time]]
    d = d.rename(columns={
        col_dec: 'decision',
        col_game: 'game_id',
        col_time: 'decision_time'
    })
    # Ordena per game_id però manté l’ordre intern (ordenació estable)
    d = d.sort_values(by='game_id', kind='mergesort')
    return d

# Creem els quatre subdatasets
df_A = prepara_story(df, 'A', 'decision_A', 'game_id_A', 'decision_time_A')
df_B = prepara_story(df, 'B', 'decision_B', 'game_id_B', 'decision_time_B')
df_C = prepara_story(df, 'C', 'decision_C', 'game_id_C', 'decision_time_C')
df_D = prepara_story(df, 'D', 'decision_D', 'game_id_D', 'decision_time_D')


df_depurat = pd.concat([df_A, df_B, df_C, df_D], ignore_index='true')
df_depurat=df_depurat[df_depurat['decision'].notna()]


n_files = df_depurat.shape[0]
pattern = np.tile(np.arange(1, 7), n_files // 6 + 1)[:n_files]
df_1 = df_depurat.copy()
df_1['ordre'] = pattern
df_1['age'] = df_1['age'].replace(['r1','r2'],'Young')
df_1['age'] = df_1['age'].replace(['r3','r4', 'r5'],'Adult')
df_1['age'] = df_1['age'].replace(['r6','r7','r8'],'Old')

df_2 = df_depurat[df_depurat['age'].isin(['r1', 'r2'])]
df_3 = df_depurat[df_depurat['age'].isin(['r3', 'r4', 'r5'])]
df_4 = df_depurat[df_depurat['age'].isin(['r6', 'r7', 'r8'])]
df_5 = df_depurat[df_depurat['gender'].isin(['W'])]
df_6 = df_depurat[df_depurat['gender'].isin(['M'])]
df_7 = df_depurat[df_depurat['decision'].isin(['C'])]
df_8 = df_depurat[df_depurat['decision'].isin(['I'])]
df_9 = df_depurat[df_depurat['decision'].isin(['D'])]
df_10 = df_depurat[df_depurat['story'].isin(['A'])]
df_11 = df_depurat[df_depurat['story'].isin(['B'])]
df_12 = df_depurat[df_depurat['story'].isin(['C'])]
df_13 = df_depurat[df_depurat['story'].isin(['D'])]


dfs = [df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12,df_13]


dades, valors_q, n_q, n_n = arrays(dfs,0,20.1,0.1)


moms, ln_moms = moments(dades, valors_q, n_q, n_n )

etiquetes = ['Total', 'Young', 'Adult', 'Old', 'Women', 'Man', 'C', 'I', 'D','A','B','C','D']
print('Temps mitjà intervent time i error')
for i in range (len(etiquetes)):
    mit = moms[np.where(valors_q == 1.0)[0],i]
    var = np.average(((dades[i][:,6])-mit)**2)
    print(etiquetes[i],round(mit[0],np.sqrt(var/dades[i].shape[0])))
    
def temps_mitja_condicions(dades,condicio1,condicio2):
    genere = ['W', 'M']
    edat = ['Adult', 'Old',  'Young']
    decisio = ['I', 'C', 'D']
    suma = []
    if condicio1 in genere:
        ix1 = 0
    if condicio1 in edat:
        ix1 = 1
    if condicio1 in decisio:
        ix1 = 3
    if condicio2 in genere:
        ix2 = 0
    if condicio2 in edat:
        ix2 = 1
    if condicio2 in decisio:
        ix2 = 3
    for i in range (0,len(dades)):
        if dades[i,ix1] == condicio1 and dades[i,ix2] == condicio2:
            suma.append(dades[i,6])
    suma_arr = np.array(suma)
    return round(np.average(suma_arr),np.std(suma_arr)/np.sqrt(np.size(suma_arr)))

grups = ['Young', 'Adult', 'Old', 'W', 'M', 'C', 'I', 'D']
matriu_mitjanes = np.empty([len(grups),len(grups)]).astype('object')
for i in range (0,len(grups)):
    for j in range (0,len(grups)):
        matriu_mitjanes[i,j] = temps_mitja_condicions(dades[0], grups[i], grups[j])
print(matriu_mitjanes)




#%%
# --- OPERACIONS AMB EL TEMPS VARIES---

tau, tau1 = taus_1(dades[0],6)

#aquí construïm temps desde 0 per a cada joc
x = dades[0][:,6].copy()
t_max=[]
grups = 6
blocs = [x[i:i+grups] for i in range(0, len(x), grups)]

#sumem temps dividint per jocs
for array in blocs:
    for i in range (1,len(array)):
        array[i] = array[i]+array[i-1]
        #if i % 5 == 0 and i !=0:
            #t_max.append(array[i])
t = np.concatenate(blocs)

#temps per a cada participant
dades_t = dades[0].copy()
dades_t[:,6] = t

#generam una distribució aletoria per a comparar. Hem de tenir en compte que 
#així traiem el t, no el intervent time. Com que tenim poques mostres en 
#generem bastantes i despres fem mitajnes

def generador_poisson(dades,samples,grups):
    tau_poiaux = []
    llargada = dades.shape[0]
    divisions = int(llargada/grups)
    for k in range (0,samples):
        t_poiaux = np.zeros((llargada,samples))
        for i in range (0,llargada):
            t_poiaux[i,k] = random.uniform(0,samples)
        
        construir_tau = t_poiaux.copy()
        blocs = [construir_tau[i:i+grups,k] for i in range(0, len(construir_tau[:,0]), grups)]
        for array in blocs:
            array.sort()
            for i in range (1, len(array)):
                array[i] = array[i]-array[i-1]
    
        tau_poiaux.append(np.concatenate(blocs))
    t_poi = np.average(t_poiaux, axis=0)
    tau_poi = np.average(np.array(tau_poiaux), axis=0)
    return t_poi, tau_poi

t_poi, tau_poi = generador_poisson(dades[0][:,6],1,grups)

#reescalem les dades de poisson perque tinguin la mitjana apropiada
coef = np.average(tau_poi)/np.average(dades[0][:,6])
tau_poi = tau_poi/coef
print(np.average(tau_poi))

x = tau1.copy()
y = tau.copy()
# Substituïm zeros per NaN
x = np.where(x == 0, np.nan, x)

plt.figure()
# Separem dades per sexe
mask_woman = y[:,0] == 'W'
mask_man = ~mask_woman
plt.figure()
plt.plot(x[:,6][mask_woman], y[:,6][mask_woman], 'o', markersize=3, label='Woman')
plt.plot(x[:,6][mask_man], y[:,6][mask_man], 'o', markersize=3, label='Man')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_sexe.png', dpi=300, bbox_inches='tight')
plt.close()
    
plt.figure()
# Separem dades per edat
mask_young = y[:,1] == 'Young'
mask_adult = y[:,1] == 'Adult'
mask_old = y[:,1] == 'Old'
plt.figure()
plt.plot(x[:,6][mask_young], y[:,6][mask_young], 'o', markersize=3, label='Young')
plt.plot(x[:,6][mask_adult], y[:,6][mask_adult], 'o', markersize=3, label='Adult')
plt.plot(x[:,6][mask_old], y[:,6][mask_old], 'o', markersize=3, label='Old')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_edat.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
# Separem dades per edat
mask_I = y[:,3] == 'I'
mask_C = y[:,3] == 'C'
mask_D = y[:,3] == 'D'
plt.figure()
plt.plot(x[:,6][mask_I], y[:,6][mask_I], 'o', markersize=3, label='I')
plt.plot(x[:,6][mask_C], y[:,6][mask_C], 'o', markersize=3, label='C')
plt.plot(x[:,6][mask_D], y[:,6][mask_D], 'o', markersize=3, label='D')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_decisio.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
# Separem dades per historia
mask_A = y[:,4] == 'A'
mask_B = y[:,4] == 'B'
mask_C = y[:,4] == 'C'
mask_D = y[:,4] == 'D'
plt.figure()
plt.plot(x[:,6][mask_A], y[:,6][mask_A], 'o', markersize=3, label='A')
plt.plot(x[:,6][mask_B], y[:,6][mask_B], 'o', markersize=3, label='B')
plt.plot(x[:,6][mask_C], y[:,6][mask_C], 'o', markersize=3, label='C')
plt.plot(x[:,6][mask_D], y[:,6][mask_D], 'o', markersize=3, label='D')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_situacio.png', dpi=300, bbox_inches='tight')
plt.close()

#scatter poisson i experimentals
tau_p, tau1_p = taus_1(tau_poi,6)
tau1_p = np.where(tau1_p == 0, np.nan, tau1_p)
tau1 = np.where(tau1 == 0, np.nan, tau1)
plt.figure()
plt.plot(tau1[:,6], tau[:,6], 'o', markersize=3, label='Experimental')
plt.plot(tau1_p, tau_p, 'o', markersize=3, label='Poisson')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.xlim(0,None)
plt.ylim(0,None)
plt.legend(frameon=False)
#plt.savefig('scatter_exp_poisson.png', dpi=300, bbox_inches='tight')
plt.close()

#mitjanes de Poisson i dades experiemntals
datasets = 2
juntar_data = [dades[0][:,6], tau_poi]
valors = np.arange(0,np.max(dades[0][:,6]),0.1)
mitjanes = np.zeros([valors.shape[0],datasets])
for j in range (0,datasets):
    for i, v in enumerate(valors):
          mitjanes[i,j] = mitjana_condi(juntar_data[j],6,v)
eti_mit = ['Experimental','Poisson']

mitjanes_lokes = mitjanes

plt.figure()
for i in range (0,datasets):
    plt.plot(valors,mitjanes[:,i],'o',markersize=2,label=eti_mit[i])
    #plt.plot(model_mitjanes(x, a, b, c),'-')
# plt.xlim(2.4,np.max(valors)+1)
# plt.ylim(7,14)
# plt.xticks(np.arange(0,np.max(valors),5))
# plt.yticks(np.arange(7,16,1))
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$\tau_{i-1} \ (s)$')
plt.ylabel(r'$\langle \tau_i| \tau_{i-1} \rangle \ (s)$')
plt.savefig('mitjanes_exp.png', dpi=300, bbox_inches='tight')
plt.close()


# #Mirem la energy distance i el p-value per veure si hi ha diferencies entre les distribucions

# X = np.array(dades[0][:,6]).copy()
# X = X.reshape(-1, 1)
# Y = np.array(tau_poi).copy()
# Y = Y.reshape(-1, 1)
# result = dcor.homogeneity.energy_test(X/max(X), Y/max(Y), num_resamples=1000)
# print('\n p-value per a la Experimental i Poisson')
# print(result)

# X = np.array(dades[1][:,6]).copy()
# X = X.reshape(-1, 1)
# Y = np.array(dades[2][:,6]).copy()
# Y = Y.reshape(-1, 1)
# Z = np.array(dades[3][:,6]).copy()
# Z = Z.reshape(-1, 1)
# result = dcor.homogeneity.energy_test(X/max(X), Y/max(Y), Z/max(Z),num_resamples=1000)
# print('\n p-value per a Young, Adult, Old')
# print(result)

# X = np.array(dades[4][:,6]).copy()
# X = X.reshape(-1, 1)
# Y = np.array(dades[5][:,6]).copy()
# Y = Y.reshape(-1, 1)
# result = dcor.homogeneity.energy_test(X/max(X), Y/max(Y),num_resamples=1000)
# print('\n p-value per a Woman, Man')
# print(result)

# X = np.array(dades[6][:,6]).copy()
# X = X.reshape(-1, 1)
# Y = np.array(dades[7][:,6]).copy()
# Y = Y.reshape(-1, 1)
# Z = np.array(dades[8][:,6]).copy()
# Z = Z.reshape(-1, 1)
# result = dcor.homogeneity.energy_test(X/max(X), Y/max(Y), Z/max(Z),num_resamples=1000)
# print('\n p-value per a I, C, D')
# print(result)

# X = np.array(dades[9][:,6]).copy()
# X = X.reshape(-1, 1)
# Y = np.array(dades[10][:,6]).copy()
# Y = Y.reshape(-1, 1)
# Z = np.array(dades[11][:,6]).copy()
# Z = Z.reshape(-1, 1)
# Ç = np.array(dades[12][:,6]).copy()
# Ç = Ç.reshape(-1, 1)
# result = dcor.homogeneity.energy_test(X/max(X), Y/max(Y), Z/max(Z), Ç/max(Ç), num_resamples=1000)
# print('\n p-value per a A, B, C, D')
# print(result)

#%%

# ---- DIVERGENCIA JS ----
dades_comulada = []
for i in range (0,len(dades)):
    dades_comulada.append(dades[i][:,6])
    

pas = 0.5
def estimador_densitat(dades,pas):
    #pas que fem desde 0 fins valor màxim
    llista_i = []
    densitat_i = []
    maxims = [arr.max() for arr in dades]
    #ens quedem ab el màxim dels màxims per a que tot tingui mateixa mida
    index = np.argmax(maxims)
    for element in dades:
        recorrer = np.arange(0,np.max(dades[index])+pas,pas)
        llista_aux = []
        for k in range (0,len(recorrer)):
            i = sum(1 for x in element if x <= recorrer[k])
            llista_aux.append(i)
        llista_i.append(np.array(llista_aux)/max(llista_aux))
    for i in range (0,len(dades)):
        densitat_i.append(gaussian_filter1d(np.gradient(llista_i[i], pas),sigma=2))
    return densitat_i

densitat = estimador_densitat(dades_comulada,pas)
        

# plt.figure()
# plt.plot(acomulada1[1],densitat1)
    

def divergenciaJS(P,Q):
    DP = np.zeros(len(P))
    DQ = np.zeros(len(P))
    M = (1/2)*(P+Q)
    for i in range (0,len(P)):
        if P[i] != 0:
            DP[i] = P[i]*np.log2(P[i]/M[i])
        if Q[i] != 0:
            DQ[i] = Q[i]*np.log2(Q[i]/M[i])
    return np.sum(DP+DQ)

matriu_JS = np.zeros([len(dades),len(dades)])
for i in range (0,len(dades)):
    plt.plot(np.arange(0,42,0.5),densitat[i])
    for j in range (0,len(dades)):
        matriu_JS[i,j] = divergenciaJS(densitat[i], densitat[j])
        
    
etiquetes = ['Young', 'Adult', 'Old', 'Women', 'Men', 'C', 'I', 'D','A','B','C','D']


plt.figure()
seaborn.heatmap(np.sqrt(matriu_JS[1:13,1:13]), vmin=0, vmax=1, cmap='viridis',
            xticklabels=etiquetes, yticklabels=etiquetes,annot = True,
            annot_kws={"fontsize":7, "color":"white"},fmt = '.2f', square=True)
plt.savefig('heatmap_distJS.png', dpi=300, bbox_inches='tight')

N = 10000
def pvalue(ix1,ix2,N):
    dades1 = dades[ix1][:,6]
    dades2 = dades[ix2][:,6]
    #fem un pool amb els dos grups
    dades_junt = np.concatenate([dades1,dades2])
    cont = 0
    for _ in range (0,N):
        #dividim les dades barrejades en 2 i calulem DivJS
        np.random.shuffle(dades_junt)
        d1 = np.zeros(dades1.shape)
        d2 = np.zeros(dades2.shape)
        d1 = dades_junt[0:len(d1)]
        d2 = dades_junt[len(d1):]
        densitat = estimador_densitat([d1,d2],pas)
        div = divergenciaJS(densitat[0], densitat[1])
        if div >= matriu_JS[ix1,ix2]:
            cont += 1
    p = cont/N
    return p

# matriu_p = []
# for ix2 in range(1,len(dades)):
#     p_fila = []
#     for ix1 in range(1,len(dades)):
#         if ix2 > ix1:
#             p_fila.append(pvalue(ix1,ix2,N))
#         else:
#             p_fila.append(np.nan)
#     matriu_p.append(p_fila)

#%%
#---BARREGEM LES DADES----

def barrejar(dades):
    # Generem número de 0 a len(dataset) i barregem
    arr = list(range(len(dades[:,6])))
    random.shuffle(arr)     
    dades_barrejades = np.zeros(dades.shape,dtype='object')
    for i in range (0,len(dades)):
        dades_barrejades[i,:] = dades[arr[i],:]
    return dades_barrejades

dades_barrejades = barrejar(dades[0])

#Fem el fet abans però per a les dades barrejades
tau_b, tau1_b = taus_1(dades_barrejades,6)
x = tau1_b[:,6].copy()
y = tau_b.copy()
# Substituïm zeros per NaN
x = np.where(x == 0, np.nan, x)

plt.figure()
# Separem dades per sexe
mask_woman = y[:,0] == 'W'
mask_man = ~mask_woman
plt.figure()
plt.plot(x[mask_woman], y[:,6][mask_woman], 'o', markersize=3, label='Woman')
plt.plot(x[mask_man], y[:,6][mask_man], 'o', markersize=3, label='Man')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_sexe_barrejat.png', dpi=300, bbox_inches='tight')
plt.close()
    
plt.figure()
# Separem dades per edat
mask_young = y[:,1] == 'Young'
mask_adult = y[:,1] == 'Adult'
mask_old = y[:,1] == 'Old'
plt.figure()
plt.plot(x[mask_young], y[:,6][mask_young], 'o', markersize=3, label='Young')
plt.plot(x[mask_adult], y[:,6][mask_adult], 'o', markersize=3, label='Adult')
plt.plot(x[mask_old], y[:,6][mask_old], 'o', markersize=3, label='Old')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_edat_barrejades.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
# Separem dades per edat
mask_I = y[:,3] == 'I'
mask_C = y[:,3] == 'C'
mask_D = y[:,3] == 'D'
plt.figure()
plt.plot(x[mask_I], y[:,6][mask_I], 'o', markersize=3, label='I')
plt.plot(x[mask_C], y[:,6][mask_C], 'o', markersize=3, label='C')
plt.plot(x[mask_D], y[:,6][mask_D], 'o', markersize=3, label='D')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_decisio_barrejades.png', dpi=300, bbox_inches='tight')
plt.close()

# Separem dades per historia
mask_A = y[:,4] == 'A'
mask_B = y[:,4] == 'B'
mask_C = y[:,4] == 'C'
mask_D = y[:,4] == 'D'
plt.figure()
plt.plot(x[mask_A], y[:,6][mask_A], 'o', markersize=3, label='A')
plt.plot(x[mask_B], y[:,6][mask_B], 'o', markersize=3, label='B')
plt.plot(x[mask_C], y[:,6][mask_C], 'o', markersize=3, label='C')
plt.plot(x[mask_D], y[:,6][mask_D], 'o', markersize=3, label='D')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.legend(frameon=False, loc='upper left')
#plt.savefig('scatter_situacio_barrejades.png', dpi=300, bbox_inches='tight')
plt.close()

#%%
#---PROBABILITATS I CHI-QUADRAT

#ho fem així ja que lúltim de cada joc mai participa
def conta_sense_ultim(dades):
    total_I = 0
    total_C = 0
    total_D = 0
    for i in range (0, len(dades[:,3])):
        if i % 6 != 0:
            if dades[i-1,3] == 'I':
                total_I += 1
            elif dades[i-1,3] == 'C':
                total_C += 1
            elif dades[i-1,3] == 'D':
                total_D += 1
    total = [total_C, total_I, total_D]
    return total
            
total = conta_sense_ultim(dades[0])


#condicio es la decisio anterior i index és la columna que volem comparar (genera, edat, etc)
def decisio_amb_anterior(dades,index,condicio):
    contador_I = 0
    contador_C = 0
    contador_D = 0
    temps_I = []
    temps_C = []
    temps_D = []    
    for i in range(1,len(dades[:,6])):
        #així no comparem entre jocs
        if i % 6 != 0:
            if (dades[i,index] == condicio and dades[i-1,3] == 'C'):
                contador_C += 1
                temps_C.append(dades[i,6])
            elif (dades[i,index] == condicio and dades[i-1,3] == 'I'):
                contador_I += 1
                temps_I.append(dades[i,6])
            elif (dades[i,index] == condicio and dades[i-1,3] == 'D'):
                contador_D += 1
                temps_D.append(dades[i,6])
    return [temps_C, temps_I, temps_D], [contador_C, contador_I, contador_D]

temps_perC, contadors_perC = decisio_amb_anterior(dades[0], 3,'C')
temps_perI, contadors_perI = decisio_amb_anterior(dades[0], 3,'I')
temps_perD, contadors_perD = decisio_amb_anterior(dades[0], 3,'D')

#probabilitat de contadors_per les condicions
def prob_condi(contador,total):
    prob = []
    for i in range (0,3):
        prob.append(contador[i]/(total[i]))
    return prob

prob_I = prob_condi(contadors_perI,total)
prob_C = prob_condi(contadors_perC,total)
prob_D = prob_condi(contadors_perD,total)

print('\n Probabilitat de tenir C donat C, I o D')
print(prob_C)

print('\n Probabilitat de tenir I donat C, I o D')
print(prob_I)

print('\n Probabilitat de tenir D donat C, I o D')
print(prob_D)

def decisio_amb_actual(dades,index,condicio):
    contador_Y = 0
    contador_A = 0
    contador_O = 0  
    for i in range(1,len(dades[:,6])):
        if (dades[i,3] == condicio and dades[i,index] == 'Young'):
            contador_Y += 1
        elif (dades[i,3] == condicio and dades[i,index] == 'Adult'):
            contador_A += 1
        elif (dades[i,3] == condicio and dades[i,index] == 'Old'):
            contador_O += 1
    return [contador_Y, contador_A, contador_O]


#codi com lo d'abans però amb petit canvis
totalC = 0
totalI = 0
totalD = 0
for i in range(0, len(dades[0][:,6])):
    if dades[0][i,1] == 'Young':
        totalC += 1
    elif dades[0][i,1] == 'Adult':
        totalI += 1
    elif dades[0][i,1] == 'Old':
        totalD += 1
total_age = [totalC,totalI,totalD]

contadors_perC_a = decisio_amb_actual(dades[0], 1,'C')
contadors_perI_a = decisio_amb_actual(dades[0], 1,'I')
contadors_perD_a = decisio_amb_actual(dades[0], 1,'D')

#probabilitat de C per a Young, Adult i Old
prob_C_a = prob_condi(contadors_perC_a,total_age)
prob_I_a = prob_condi(contadors_perI_a,total_age)
prob_D_a = prob_condi(contadors_perD_a,total_age)

print('\n Probabilitat de tenir C donat Y, A o O')
print(prob_C_a)

print('\n Probabilitat de tenir I donat Y, A o O')
print(prob_I_a)

print('\n Probabilitat de tenir D donat Y, A o O')
print(prob_D_a)

def decisio_segons_joc(dades,index,condicio):
    contador_A = 0
    contador_B = 0
    contador_C = 0  
    contador_D = 0  
    for i in range(1,len(dades[:,6])):
        if (dades[i,3] == condicio and dades[i,index] == 'Young'):
            contador_Y += 1
        elif (dades[i,3] == condicio and dades[i,index] == 'Adult'):
            contador_A += 1
        elif (dades[i,3] == condicio and dades[i,index] == 'Old'):
            contador_O += 1
    return [contador_Y, contador_A, contador_O]

#dades és una llista
def mitjana(dades):
    if len(dades) == 0:
        m = 0
    else:   
        m = sum(dades)/len(dades)
    return m

def var(dades):
    var = []
    for  i in range (0,len(dades)):
        var.append((dades[i]-mitjana(dades))**2)
    v = mitjana(var)  
    return v

def incert(dades):
    if len(dades) == 0:
        return 0
    else:
        return np.sqrt(var(dades)/len(dades))

print('\n Mitjana de temps C donat C, I o D')
print(round(mitjana(temps_perC[0]),incert(temps_perC[0])),round(mitjana(temps_perC[1]),incert(temps_perC[1])),round(mitjana(temps_perC[2]),incert(temps_perC[2])))
print('\n Mitjana de temps I donat C, I o D')
print(round(mitjana(temps_perI[0]),incert(temps_perI[0])),round(mitjana(temps_perI[1]),incert(temps_perI[1])),round(mitjana(temps_perI[2]),incert(temps_perI[2])))
print('\n Mitjana de temps D donat C, I o D')
print(round(mitjana(temps_perD[0]),incert(temps_perD[0])),round(mitjana(temps_perD[1]),incert(temps_perD[1])),round(mitjana(temps_perD[2]),incert(temps_perD[2])))


#fem el mateix per a les barrejades
total_b = conta_sense_ultim(dades_barrejades)
temps_perI, contadors_perI = decisio_amb_anterior(dades_barrejades, 3,'I')
temps_perC, contadors_perC = decisio_amb_anterior(dades_barrejades, 3,'C')
temps_perD, contadors_perD = decisio_amb_anterior(dades_barrejades, 3,'D')
prob_I = prob_condi(contadors_perI,total_b)
prob_C = prob_condi(contadors_perC,total_b)
prob_D = prob_condi(contadors_perD,total_b)
print('\n Probabilitat de tenir C donat C, I o D per a les barrejades')
print(prob_C)
print('\n Probabilitat de tenir I donat C, I o D per a les barrejades')
print(prob_I)
print('\n Probabilitat de tenir D donat C, I o D per a les barrejades')
print(prob_D)
print('\n Mitjana de temps C donat C, I o D per a les barrejades')
print(round(mitjana(temps_perC[0]),incert(temps_perC[0])),round(mitjana(temps_perC[1]),incert(temps_perC[1])),round(mitjana(temps_perC[2]),incert(temps_perC[2])))
print('\n Mitjana de temps I donat C, I o D per a les barrejades')
print(round(mitjana(temps_perI[0]),incert(temps_perI[0])),round(mitjana(temps_perI[1]),incert(temps_perI[1])),round(mitjana(temps_perI[2]),incert(temps_perI[2])))
print('\n Mitjana de temps D donat C, I o D per a les barrejades')
print(round(mitjana(temps_perD[0]),incert(temps_perD[0])),round(mitjana(temps_perD[1]),incert(temps_perD[1])),round(mitjana(temps_perD[2]),incert(temps_perD[2])))

contingencia_ordre = pd.crosstab(df_1['ordre'], df_1['decision'])
contingencia_edat = pd.crosstab(df_1['age'], df_1['decision'])


def cramers_v(taula):
    chi2, p, dof, expected = chi2_contingency(taula)
    n = taula.to_numpy().sum()
    k = min(taula.shape)
    return chi2, p, np.sqrt(chi2 / (n * (k - 1)))

chi2_ordre, p_ordre, V_ordre = cramers_v(contingencia_ordre)
chi2_edat, p_edat, V_edat = cramers_v(contingencia_edat)

print('\n chi-quadrat per a ordre ')
print(chi2_ordre, p_ordre, V_ordre)

print('\n chi-quadrat per a edat')
print(chi2_edat, p_edat, V_edat)




#%%
#scatter poisson i barrejades
plt.figure()
plt.plot(tau1_b[:,6], tau_b[:,6], 'o', markersize=3, label='Barrejades')
plt.plot(tau1_p, tau_p, 'o', markersize=3, label='Poisson')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$ \tau_{i-1} $ (s)')
plt.ylabel(r'$ \tau_{i} $ (s)')
plt.xlim(0,None)
plt.ylim(0,None)
plt.legend(frameon=False)
#plt.savefig('scatter_barrejades_poisson.png', dpi=300, bbox_inches='tight')
plt.close()

#mitjanes de Poisson i dades barrejades
datasets = 2
juntar_data = [dades_barrejades[:,6], tau_poi]
valors = np.arange(0,np.max(dades[0][:,6]),0.1)
mitjanes = np.zeros([valors.shape[0],datasets])
for j in range (0,datasets):
    for i, v in enumerate(valors):
          mitjanes[i,j] = mitjana_condi(juntar_data[j],6,v)
          #print(mitjanes[i,j])
eti_mit = ['Barrejades','Poisson']
plt.figure()
for i in range (0,datasets):
    plt.plot(valors,mitjanes[:,i],'o',markersize=2,label=eti_mit[i])
    #plt.plot(model_mitjanes(x, a, b, c),'-')
plt.xlim(2.4,np.max(valors)+1)
plt.ylim(7,14)
# plt.xticks(np.arange(0,np.max(valors),5))
# plt.yticks(np.arange(7,16,1))
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$\tau_{i-1} \ (s)$')
plt.ylabel(r'$\langle \tau_i| \tau_{i-1} \rangle \ (s)$')
plt.savefig(f'mitjanes_barrejades.png', dpi=300, bbox_inches='tight')
plt.close()

juntar_data = [dades[0][:,6],dades_barrejades[:,6]]
valors = np.arange(0,np.max(dades[0][:,6]),0.1)
mitjanes = np.zeros([valors.shape[0],datasets])
for j in range (0,datasets):
    for i, v in enumerate(valors):
          mitjanes[i,j] = mitjana_condi(juntar_data[j],6,v)
          #print(mitjanes[i,j])
eti_mit = ['Experimental','Barrejades']
plt.figure()
for i in range (0,datasets):
    plt.plot(valors,mitjanes[:,i],'o',markersize=2,label=eti_mit[i])
    #plt.plot(model_mitjanes(x, a, b, c),'-')
plt.xlim(2.4,np.max(valors)+1)
plt.ylim(7,None)
# plt.xticks(np.arange(0,np.max(valors),5))
# plt.yticks(np.arange(7,16,1))
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$\tau_{i-1} \ (s)$')
plt.ylabel(r'$\langle \tau_i| \tau_{i-1} \rangle \ (s)$')
plt.savefig(f'mitjanes_exp_barr.png', dpi=300, bbox_inches='tight')
plt.close()

X = np.array(dades_barrejades[:,6]).copy()
X = X.reshape(-1, 1)
Y = np.array(tau_poi).copy()
Y = Y.reshape(-1, 1)
result = dcor.homogeneity.energy_test(X, Y, num_resamples=1000)
#%%

#---DECOMULADES---

#aquí construïm temps desde 0 per a cada joc a les dades barrejades
x = dades_barrejades[:,6].copy()
grups = 6
blocs = [x[i:i+grups] for i in range(0, len(x), grups)]
#sumem temps dividint per jocs
for array in blocs:
    for i in range (1,len(array)):
        array[i] = array[i]+array[i-1]
t_b = np.concatenate(blocs)


def dis_decomulada(dades):
    llista_i = []
    maxims = [arr.max() for arr in dades]
    index = np.argmax(maxims)
    for element in dades:
        recorrer = np.arange(0,np.max(dades[index])+0.1,0.1)
        llista_aux = []
        for k in range (0,len(recorrer)):
            i = sum(1 for x in element if x >= recorrer[k])
            llista_aux.append(i)
        llista_i.append(np.array(llista_aux)/max(llista_aux))
    return llista_i, recorrer

llista = [np.sort(tau[:,6]),np.sort(tau_poi),np.sort(t),np.sort(t_poi)]
llista_i, recorrer = dis_decomulada(llista)
    
etiquetes = [r'$\tau_e$', r'Poisson']#, r'$t_e$', r'$t_p$']

def decomulada(x,y,etiquetes,nom,guardar):
    plt.figure()
    datasets = len(etiquetes)
    for i in range (0, 1):
        y[i] = np.where(y[i] == 0, np.nan, y[i])
        plt.plot(x,y[i],'o',markersize=2,label=etiquetes[i])
        plt.plot(x,(np.exp(-x/np.average(dades[0][:,6]))),'--',label='Poisson')
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
    plt.ylim(0,1)
    plt.xlim(0,45)
    plt.xlabel(r'${\tau} (s)$')
    plt.ylabel(r'$\Psi (t)$')
    plt.legend(frameon=False)
    if guardar == True:
        plt.savefig(f'{nom}.png', dpi=300, bbox_inches='tight')
    plt.close()

decomulada(recorrer,llista_i,etiquetes,'decomulada interevent',True)

#Ajuts lineal
def regressio(x, y, start_idx, qmin, qmax):
    # Seleccionem els valors a partir de start_idx
    x = x[start_idx:]
    y = y[start_idx:]
    # Filtratge per interval
    mask_interval = (x >= qmin) & (x <= qmax)
    X = x[mask_interval]
    y_fit = y[mask_interval]
    # Afegim constant per intercept
    X_const = sm.add_constant(X)
    # Model lineal
    model = sm.OLS(y_fit, X_const).fit()
    # Paràmetres i errors
    intercept, slope = model.params
    e_intercept, e_slope = model.bse

    return intercept, slope, e_intercept, e_slope, X, y_fit

intercept, slope, e_intercept, e_slope, X, y_fit = regressio(recorrer, np.log(llista_i[0]), 0, 10, 25)
print('\n Valors per al ajust decomulada')
print(round(intercept,e_intercept), round(slope,e_slope))

#Gràfic amb l'ajjst per al ln de la decomulada
plt.figure()
y = np.log(llista_i[0]) 
x = recorrer
y = np.where(y == 0, np.nan, y)
plt.plot(x,y,'o',markersize=2,label=etiquetes[0],color='#1f77b4')
plt.plot(x,intercept+slope*x,'--',color='#1f77b4')
plt.plot(x,np.log(np.exp(-x/np.average(dades[0][:,6]))),'--',label='Poisson',color='#ff7f0e')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.ylim(-6,0)
plt.xlim(0,45)
plt.xlabel(r'${\tau} (s)$')
plt.ylabel(r'$\ln \Psi (t)$')
plt.legend(frameon=False)
plt.savefig(f'ln decomulada.png', dpi=300, bbox_inches='tight')
plt.close()



etiquetes = ['Young', 'Adult', 'Old']
llista = []
for i in range (1,1+len(etiquetes)):
    llista.append(np.sort(dades[i][:,6]/np.average(dades[i][:,6])))
llista_i, recorrer = dis_decomulada(llista)  
decomulada(recorrer,llista_i,etiquetes,'decomulada edat',False)

etiquetes = ['Women', 'Men']
llista = []
for i in range (4,4+len(etiquetes)):
    llista.append(np.sort(dades[i][:,6]/np.average(dades[i][:,6])))
llista_i, recorrer = dis_decomulada(llista)  
decomulada(recorrer,llista_i,etiquetes,'decomulada genere',False)

etiquetes = ['Decision C', 'Decision I', 'Decision D']
llista = []
for i in range (6,6+len(etiquetes)):
    llista.append(np.sort(dades[i][:,6]/np.average(dades[i][:,6])))
llista_i, recorrer = dis_decomulada(llista)  
decomulada(recorrer,llista_i,etiquetes,'decomulada decisio',True)

etiquetes = ['Situation A', 'Situation B','Situation C','Situation D']
llista = []
for i in range (9,9+len(etiquetes)):
    llista.append(np.sort(dades[i][:,6]/np.average(dades[i][:,6])))
llista_i, recorrer = dis_decomulada(llista)  
decomulada(recorrer,llista_i,etiquetes,'decomulada situacio',False)

#%%
#---GRAFICS DELS Q MOMENTS---

#Gràfic dels q moments en zoom a valors petits
plt.figure()
etiquetes = ['Total', 'Young', 'Adult', 'Old', 'Women', 'Men', 'C', 'I', 'D']
colors = plt.cm.tab10(np.linspace(0, 1, len(etiquetes)))
x = valors_q.copy()
y = ln_moms.copy()
n_dades = len(etiquetes)
# Gràfic principal
fig, ax1 = plt.subplots()
for i in range(n_dades):
    ax1.plot(x, y[:, i], 'o', markersize=1,color=colors[i], label=etiquetes[i])
ax1.set_xticks(np.linspace(0, 20, 5))
ax1.set_yticks(np.arange(0, 80, 10))
ax1.set_xlim(-0, 20)
ax1.set_ylim(0, 70)
ax1.legend(frameon=False)
ax1.tick_params(axis='both', which='both', top=True, right=True, direction='out')
ax1.set_xlabel('q')
ax1.set_ylabel(r'$\ln \langle {\tau^q} \rangle$')
# Gràfic petit (inset) per q petites
ax2 = plt.axes([0, 0, 1, 1])
ip = InsetPosition(ax1, [0.6, 0.1, 0.3, 0.3])  # [x0, y0, width, height]
ax2.set_axes_locator(ip)
for i in range(n_dades):
    ax2.plot(x, y[:, i], 'o', markersize=1, color=colors[i])
ax2.set_xlim(0, 3)
ax2.set_ylim(0, 7)
ax2.set_xticks(np.arange(0, 4, 1))
ax2.set_yticks(np.arange(0, 7, 2))
#plt.savefig('moments_plot.png', dpi=300, bbox_inches='tight')
plt.close()

intercept =[]
slope = []
e_intercept =[]
e_slope =[]
x_fit =[]
y_fit =[]

for i in range(0,len(etiquetes)):
    intercept.append(0)
    slope.append(0)
    e_intercept.append(0)
    e_slope.append(0)
    x_fit.append(0)
    y_fit.append(0)
    
print('\n Origen, pendent, error origen i error pendent per les dades experimentals')
for i in range(0,len(etiquetes)):
    intercept[i], slope[i], e_intercept[i], e_slope[i], x_fit[i], y_fit[i] = regressio(valors_q,ln_moms[:,i],0,5.,20.)
    print(etiquetes[i],round(intercept[i],e_intercept[i]), round(slope[i],e_slope[i]))
    
#Gràfic amb l'ajust lineal tirat    
plt.figure()
y = ln_moms.copy()
x = valors_q.copy()
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=0.5, color=colors[i],label=etiquetes[i])
    plt.plot(x_fit[i], slope[i]*x_fit[i]+intercept[i], '-',color=colors[i] )
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
plt.savefig('moments_plot_ajust_lineal.png', dpi=300, bbox_inches='tight')
plt.close()

#fem ajustos multifactals, mateix codi si canviem la funcio ens erveix
def ajust_multifractal(x,A,b,c):
    y = x*A + b*x**c
    return y

popt = []
pcov = []

y = ln_moms.copy()
x = valors_q.copy()
x_mask = np.zeros(x.shape)
x_mask = (x <= 1.9) & (x>=0.0)
x_fit = x[x_mask]
y_fit = y[x_mask]

for i in range (0, len(etiquetes)):
    popt_i, pcov_i = curve_fit(ajust_multifractal, x_fit, y_fit[:,i], maxfev=10000)
    popt.append(popt_i)
    pcov.append(pcov_i)
    sigma = np.diag(pcov[i])

print('\n a, b, b origen i error per les dades experimentals')
for i in range (0, len(etiquetes)):
    sigma = np.sqrt(np.diag(pcov[i]))
    a = popt[i][0]
    b = popt[i][1]
    c = popt[i][2]
    print(round(a,sigma[0]), round(b,sigma[1]), round(c,sigma[2]))
    
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=0.5, color=colors[i],label=etiquetes[i])
    plt.plot(x, ajust_multifractal(x,popt[i][0],popt[i][1],popt[i][2]), '-', color=colors[i])
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle}$')
plt.savefig('ajust_multifractal.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#ajust heurístic per a les dades
def ajust_heuristic(x,a,b1,b,c):
    y = x*a+(1/b1)*(1-np.exp(-b1*x**(c)))*b*x
    return y

popt = []
pcov = []
for i in range (0, len(etiquetes)):
    popt_i, pcov_i = curve_fit(ajust_heuristic, valors_q, ln_moms[:,i], maxfev=10000000)
    popt.append(popt_i)
    pcov.append(pcov_i)

y = ln_moms.copy()
x = valors_q.copy()

plt.figure()
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=0.5, color=colors[i],label=etiquetes[i])
    plt.plot(x, ajust_heuristic(x,popt[i][0],popt[i][1],popt[i][2],popt[i][3]), '-', color=colors[i])
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
plt.savefig('moments_ajust_heuristic.png', dpi=300, bbox_inches='tight')
plt.close()

a = popt[:][0]
b1 = popt[:][1]
b = popt[:][2]
c = popt[:][3]

plt.figure()
print('\n a, b1, b, alfa i errors per a pràmetres ajust heurístic dades experimentals')
for i in range(0,len(etiquetes)):
    sigma = np.sqrt(np.diag(pcov[i]))
    print(etiquetes[i], round(popt[i][0], sigma[0]),round(popt[i][1],sigma[1]),round(popt[i][2], sigma[2]), round(1/popt[i][3]+1,1/popt[i][3]**2*sigma[3]))
    plt.plot(popt[i][1]*valors_q**popt[i][3],popt[i][1]/popt[i][2]*((1/valors_q)*ln_moms[:,i]-np.log(popt[i][0])),'o', markersize=1, color=colors[i],label=etiquetes[i])
plt.xticks(np.arange(0,4,0.5))
plt.yticks(np.arange(0,1.1,0.2))
plt.xlim(0,4.0)
plt.ylim(0,1.0)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$b_1 q^{c}$')
plt.ylabel(r'$f_{H}$')
plt.savefig('ajust_heuristic_comprvacio.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
#%%
# # ---- Q-MOMENTS NORMALITZATS ----

# dades_norm = dades.copy()
# for i in range (0,len(etiquetes)):
#     dades_norm[i][:,6] = dades[i][:,6]/np.average(dades[i][:,6])

# moms_norm, ln_moms_norm = moments(dades_norm, valors_q, n_q, n_n )

# plt.figure()
# etiquetes = ['Total', 'Young', 'Adult', 'Old', 'Women', 'Man', 'I', 'C', 'D']
# colors = plt.cm.tab10(np.linspace(0, 1, len(etiquetes)))
# x = valors_q.copy()
# y = ln_moms_norm.copy()
# n_dades = len(etiquetes)
# # Gràfic principal
# fig, ax1 = plt.subplots()
# for i in range(n_dades):
#     ax1.plot(x, y[:, i], 'o', markersize=1,color=colors[i], label=etiquetes[i])
# ax1.set_xticks(np.linspace(0, 20, 5))
# ax1.set_yticks(np.arange(0, 80, 10))
# ax1.set_xlim(-0, 20)
# ax1.set_ylim(0, 70)
# ax1.legend(frameon=False)
# ax1.tick_params(axis='both', which='both', top=True, right=True, direction='out')
# ax1.set_xlabel('q')
# ax1.set_ylabel(r'$\ln \langle {\tau^q} \rangle$')
# # Gràfic petit (inset) per q petites
# ax2 = plt.axes([0, 0, 1, 1])
# ip = InsetPosition(ax1, [0.6, 0.1, 0.3, 0.3])  # [x0, y0, width, height]
# ax2.set_axes_locator(ip)
# for i in range(n_dades):
#     ax2.plot(x, y[:, i], 'o', markersize=1, color=colors[i])
# ax2.set_xlim(0, 3)
# ax2.set_ylim(0, 7)
# ax2.set_xticks(np.arange(0, 4, 1))
# ax2.set_yticks(np.arange(0, 7, 2))
# plt.savefig('moments_norm_plot.png', dpi=300, bbox_inches='tight')
# plt.close()

#%%
#---    Q-MOMENTSPER ORDRE DE DECISIÓ  -----
#fem q moments segons ordre de decisio
df_tau1 = []
df_tau2 = []
df_tau3 = []
df_tau4 = []
df_tau5 = []
df_tau6 = []
df_tau1_b = []
df_tau2_b = []
df_tau3_b = []
df_tau4_b = []
df_tau5_b = []
df_tau6_b = []

for i in range (0,len(dades[0][:,6])):
    if i % 6 == 0:
        df_tau1.append(dades[0][i,:])
        df_tau1_b.append(dades_barrejades[i,:])
    elif (i+1) % 6 == 0:
        df_tau6.append(dades[0][i,:])
        df_tau6_b.append(dades_barrejades[i,:])
    elif (i+2) % 6 == 0:
        df_tau5.append(dades[0][i,:])
        df_tau5_b.append(dades_barrejades[i,:])
    elif (i+3) % 6 == 0:
        df_tau4.append(dades[0][i,:])
        df_tau4_b.append(dades_barrejades[i,:])
    elif (i+4) % 6 == 0:
        df_tau3.append(dades[0][i,:])
        df_tau3_b.append(dades_barrejades[i,:])
    elif (i+5) % 6 == 0:
        df_tau2.append(dades[0][i,:])
        df_tau2_b.append(dades_barrejades[i,:])

df_tau1 = pd.DataFrame(df_tau1)
df_tau2 = pd.DataFrame(df_tau2)
df_tau3 = pd.DataFrame(df_tau3)
df_tau4 = pd.DataFrame(df_tau4)
df_tau5 = pd.DataFrame(df_tau5)
df_tau6 = pd.DataFrame(df_tau6)
df_tau1_b = pd.DataFrame(df_tau1_b)
df_tau2_b = pd.DataFrame(df_tau2_b)
df_tau3_b = pd.DataFrame(df_tau3_b)
df_tau4_b = pd.DataFrame(df_tau4_b)
df_tau5_b = pd.DataFrame(df_tau5_b)
df_tau6_b = pd.DataFrame(df_tau6_b)
    
dfs = [df_tau1,df_tau2,df_tau3,df_tau4,df_tau5,df_tau6]

dades_tau, valors_q, n_q_tau, n_n_tau = arrays(dfs,0,20.1,0.1)

moms_tau, ln_moms_tau = moments(dades_tau, valors_q, n_q_tau, n_n_tau )
etiquetes = [r'$\tau_1$',r'$\tau_2$',r'$\tau_3$',r'$\tau_4$',r'$\tau_5$',r'$\tau_6$']

print('\n Mitjanes i error dels intervent times quan organitzem per ordre intervenció')
for i in range(0,len(etiquetes)):
    print(etiquetes[i],moms_tau[np.where(valors_q == 1.0)[0],i],np.sqrt(moms_tau[np.where(valors_q == 2.0)[0],i]/dades_tau[i].shape[0]))


def corr_matrix(x,y):
    n = len(x)
    m = len(y)
    matriu_correlacio = np.zeros([n,m])
    for i in range (0,n):
        for j in range (0,m):
            matriu_correlacio[i,j] = dcor.distance_correlation(x[i][:,6],y[j][:,6])
    return matriu_correlacio
            
matriu = corr_matrix(dades_tau,dades_tau)

#Gràfic amb l'ajust lineal tirat    
plt.figure()
y = ln_moms_tau.copy()
x = valors_q.copy()
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
#plt.savefig('moments_per_ordre.png', dpi=300, bbox_inches='tight')
plt.close()

# #fem ajustos lienal per als q moments
# intercept =[]
# slope = []
# e_intercept =[]
# e_slope =[]
# x_fit =[]
# y_fit =[]

# for i in range(0,len(etiquetes)):
#     intercept.append(0)
#     slope.append(0)
#     e_intercept.append(0)
#     e_slope.append(0)
#     x_fit.append(0)
#     y_fit.append(0)
    
# for i in range(0,len(etiquetes)):
#     intercept[i], slope[i], e_intercept[i], e_slope[i], x_fit[i], y_fit[i] = regressio(valors_q,ln_moms[:,i],0,2.,20.)
#     print(etiquetes[i],intercept[i], slope[i], e_intercept[i], e_slope[i])

# #Gràfic amb l'ajust lineal tirat    
# plt.figure()
# y = ln_moms_tau.copy()
# x = valors_q.copy()
# for i in range(len(etiquetes)):
#     plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
#     plt.plot(x, slope[i]*x+intercept[i], '--', color='black')
# plt.xticks(np.linspace(0,20,5))
# plt.yticks(np.arange(0,80,10))
# plt.xlim(0,20)
# plt.ylim(0,70)
# plt.legend(frameon=False)
# plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
# plt.xlabel('q')
# plt.ylabel(r'$\ln \langle \tau^q \rangle$')
# plt.savefig('moments_plot_tau_ajust_lineal.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

#Ajust heurístic per a les dades per ordre d'intervencio
popt = []
pcov = []
for i in range (0, len(etiquetes)):
    popt_i, pcov_i = curve_fit(ajust_heuristic, valors_q, ln_moms_tau[:,i], maxfev=10000000)
    popt.append(popt_i)
    pcov.append(pcov_i)

y = ln_moms_tau.copy()
x = valors_q.copy()

for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
    plt.plot(x, ajust_heuristic(x,popt[i][0],popt[i][1],popt[i][2],popt[i][3]), '--', color='black')
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
#plt.savefig('moments_tau_ajust_heuristic.png', dpi=300, bbox_inches='tight')
plt.close()

a = popt[:][0]
b1 = popt[:][1]
b = popt[:][2]
c = popt[:][3]

print('\n Heurístic per ordre a les dades experimentals')
np.seterr(all='ignore')

for i in range(0,len(etiquetes)):
    sigma = np.sqrt(np.diag(pcov[i]))
    print(etiquetes[i], round(popt[i][0], sigma[0]),round(popt[i][1],sigma[1]),round(popt[i][2], sigma[2]), round(1/popt[i][3]+1,1/popt[i][3]**2*sigma[3]))
    plt.plot(popt[i][1]*valors_q**popt[i][3],popt[i][1]/popt[i][2]*((1/valors_q)*ln_moms_tau[:,i]-popt[i][0]),'o', markersize=1, color=colors[i],label=etiquetes[i])
plt.xticks(np.arange(0,4,0.5))
plt.yticks(np.arange(0,1.1,0.2))
plt.xlim(0,4.0)
plt.ylim(0,1.0)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$b_1 q^{c}$')
plt.ylabel(r'$f_{H}$')
#plt.savefig('ajust_heuristic_tau_comprvacio.png', dpi=300, bbox_inches='tight')
plt.close()

#ho fem per a les dades barrejades
dfs = [df_tau1_b,df_tau2_b,df_tau3_b,df_tau4_b,df_tau5_b,df_tau6_b]

dades_b, valors_q, n_q_b, n_n_b = arrays(dfs,0,20.1,0.1)

matriu_b = corr_matrix(dades_b,dades_b)

fig = plt.figure(figsize=(10,5), constrained_layout=True)

# fem un gridspec amb 2 columnes
gs = fig.add_gridspec(1, 2, width_ratios=[1,1])

# primer heatmap sense colorbar
ax1 = fig.add_subplot(gs[0,0])
seaborn.heatmap(matriu, vmin=0, vmax=1, cmap='viridis',
            xticklabels=etiquetes, yticklabels=etiquetes,
            square=True, cbar=False, ax=ax1)
ax1.set_title("Experimental", fontsize=16)
ax1.tick_params(axis='x', labelsize=16) 
ax1.tick_params(axis='y', labelsize=16) 


# segon heatmap amb colorbar petita
ax2 = fig.add_subplot(gs[0,1])
seaborn.heatmap(matriu_b, vmin=0, vmax=1, cmap='viridis',
            xticklabels=etiquetes, yticklabels=etiquetes,
            square=True, ax=ax2)
ax2.set_title("Mixed", fontsize=16)
ax2.tick_params(axis='x', labelsize=16) 
ax2.tick_params(axis='y', labelsize=16) 

plt.savefig('heatmap_correlacio.png', dpi=300, bbox_inches='tight')
plt.close()

moms_b, ln_moms_b = moments(dades_b, valors_q, n_q_b, n_n_b )
etiquetes = [r'$\tau^b_1$',r'$\tau^b_2$',r'$\tau^b_3$',r'$\tau^b_4$',r'$\tau^b_5$',r'$\tau^b_6$']

#%%

#---- TEST DE KENDALL TAU I COVARIANCIA-----
def covariancia(dades,grups):
    dades = dades.copy()
    
    x,y = taus_1(dades,grups)
    
    #una vegada fet això x i y esdevenen de 5 en x
    def mitjana(x,axis):
        mitjana = np.average(x,axis)
        return mitjana
    
    x_mit_g = mitjana(x,axis=None)
    y_mit_g = mitjana(y,axis=None)
    
    def blocs(x,grups):
        blocs = [x[i:i+grups-1] for i in range(0, len(x), grups-1)]
        return blocs
    
    #fem blocs de x
    blocs_x = blocs(x, grups)
    blocs_y = blocs(y, grups)
    
    #on guardarem les mitjanes dels blocs
    mit_b_x = np.zeros(len(blocs_x))
    mit_b_y = np.zeros(len(blocs_y))
    #les files so cada joc i les columnes la primera, segona,... intervencio
    cov_xy = np.zeros([len(blocs_y),grups-1])
    #on guardarem la coavariancia per a cada joc
    cov_k = np.zeros(len(blocs_y))
    
    for i, array in enumerate(blocs_x):
        mit_b_x[i] = mitjana(array,axis=None)
        
    for i, array in enumerate(blocs_y):
        mit_b_y[i] = mitjana(array,axis=None)
        
    for i in range(0, len(blocs_x)):
        for j in range(0, grups-1):
            cov_xy[i,j] = (blocs_x[i][j]-mit_b_x[i])*(blocs_y[i][j]-mit_b_y[i])
        #Així obtenim la coavariancia per a cada joc
        cov_k = mitjana(cov_xy,1)
            
    cov_tau = np.zeros(len(blocs_x))
    
    for i in range(0,len(blocs_x)):
        cov_tau[i] = (grups-1)*cov_k[i]+(grups-1)*(mit_b_x[i]-x_mit_g)*(mit_b_y[i]-y_mit_g)
    
    return mitjana(cov_tau,axis=None)/(np.std(x)*np.std(y))

print('\n covariancia normalitzada')
print(covariancia(dades[0][:,6],6))
print(covariancia(dades_barrejades[:,6],6))
print('\n Mitjana i error per a dades barrejades per ordre')
for i in range(0,len(etiquetes)):
    print(etiquetes[i],moms_b[np.where(valors_q == 1.0)[0],i],np.sqrt(moms_b[np.where(valors_q == 2.0)[0],i]/dades_b[i].shape[0]))


matriu_tau = np.zeros([len(dades_tau),len(dades_tau)])
matriu_ptau = matriu_tau.copy()
for i in range (0,len(dades_tau)):
    for j in range (0,len(dades_tau)):
        tauk, pk = kendalltau(dades_tau[i][:,6], dades_tau[j][:,6])
        matriu_tau[i,j] = tauk
        matriu_ptau[i,j] = pk

#%%
#Moments per ordre de decisió de dades barrejades    
plt.figure()
y = ln_moms_b.copy()
x = valors_q.copy()
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
#plt.savefig('moments_per_ordre_barrejades.png', dpi=300, bbox_inches='tight')
plt.close()

#Ajust heurístic per a les dades barrejades
popt = []
pcov = []
for i in range (0, len(etiquetes)):
    popt_i, pcov_i = curve_fit(ajust_heuristic, valors_q, ln_moms_b[:,i], maxfev=10000000)
    popt.append(popt_i)
    pcov.append(pcov_i)

y = ln_moms_b.copy()
x = valors_q.copy()

for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
    plt.plot(x, ajust_heuristic(x,popt[i][0],popt[i][1],popt[i][2],popt[i][3]), '--', color='black')
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,80,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
#plt.savefig('moments_barrejats_ajust_heuristic.png', dpi=300, bbox_inches='tight')
plt.close()

a = popt[:][0]
b1 = popt[:][1]
b = popt[:][2]
c = popt[:][3]

print('\n Heurístic per ordre a les dades barrejades')
np.seterr(all='ignore')
for i in range(0,len(etiquetes)):
    sigma = np.sqrt(np.diag(pcov[i]))
    print(etiquetes[i], round(popt[i][0], sigma[0]),round(popt[i][1],sigma[1]),round(popt[i][2], sigma[2]), round(1/popt[i][3]+1,1/popt[i][3]**2*sigma[3]))
    plt.plot(popt[i][1]*valors_q**popt[i][3],popt[i][1]/popt[i][2]*((1/valors_q)*ln_moms_b[:,i]-popt[i][0]),'o', markersize=1, color=colors[i],label=etiquetes[i])
plt.xticks(np.arange(0,4,0.5))
plt.yticks(np.arange(0,1.1,0.2))
plt.xlim(0,4.0)
plt.ylim(0,1.0)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel(r'$b_1 q^{c}$')
plt.ylabel(r'$f_{H}$')
#plt.savefig('ajust_heuristic_barrejats_comprvacio.png', dpi=300, bbox_inches='tight')
plt.close()

#decomulades per a diferents ordres
#comencem per les barrejades
llista_b = []
for i in range (0,6):
    llista_b.append(np.sort(dades_b[i][:,6]))
llista_i_b, recorrer_b = dis_decomulada(llista_b)

llista_tau = []
for i in range (0,6):
    llista_tau.append(np.sort(dades_tau[i][:,6]))
llista_i_tau, recorrer_tau = dis_decomulada(llista_tau)
    
etiquetes = [r'$\tau^b_1$',r'$\tau^b_2$',r'$\tau^b_3$',r'$\tau^b_4$',r'$\tau^b_5$',r'$\tau^b_6$']
decomulada(recorrer_b,llista_i_b,etiquetes,'decomulada_barrejades_ordre',False)

etiquetes = [r'$\tau_1$',r'$\tau_2$',r'$\tau_3$',r'$\tau_4$',r'$\tau_5$',r'$\tau_6$']
decomulada(recorrer_tau,llista_i_tau,etiquetes,'decomulada_ordre',False)

for i in range(1,7):
    etiquetes = [fr'$\tau_{i}$',fr'$\tau^b_{i}$']
    llista_i = [llista_i_tau[i-1],llista_i_b[i-1]]
    decomulada(recorrer_tau,llista_i,etiquetes,f'decomulada_ordre_barrejada_{i}',False)
    

#decomulades per a temps
#sumem temps dividint per jocs
for array in blocs:
    for i in range (1,len(array)):
        array[i] = array[i]+array[i-1]
        if i % 5 == 0 and i !=0:
            t_max.append(array[i])
t = np.concatenate(blocs)


#%%
#--- DADES AMB LA CORRECCIÓ DE GAMMA(Q+1)-----

#Corregim per gamma de q+1
def moments_corr(dades, valors_q, n_q, n_n ):
    #Càlcul de moments
    moms = np.zeros([n_q, n_n])
    for i in range (0,n_n):
        for j, q in enumerate(valors_q):
            moms[j,i] = np.mean(dades[i][:,6]**q)/gamma(q+1)
    ln_moments= np.log(moms)
    return moms, ln_moments

moms_corr, ln_moms_corr = moments_corr(dades, valors_q, n_q, n_n)

etiquetes = ['Total', 'Young', 'Adult', 'Old', 'Women', 'Man', 'I', 'C', 'D']

#plot de moments corregits
plt.figure()
y = ln_moms_corr.copy()
x = valors_q.copy()
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,32,5))
plt.xlim(0,20)
plt.ylim(0,30)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \frac{\langle \tau^q \rangle}{\Gamma (q+1)}$')
#plt.savefig('moments_gamma.png', dpi=300, bbox_inches='tight')
plt.close()

intercept =[]
slope = []
e_intercept =[]
e_slope =[]
x_fit =[]
y_fit =[]

for i in range(0,len(etiquetes)):
    intercept.append(0)
    slope.append(0)
    e_intercept.append(0)
    e_slope.append(0)
    x_fit.append(0)
    y_fit.append(0)
    
print('\n Ajust lineal amb mateix ordre que sempre per a dades corregides per gamma')
for i in range(0,len(etiquetes)):
    intercept[i], slope[i], e_intercept[i], e_slope[i], x_fit[i], y_fit[i] = regressio(valors_q,ln_moms_corr[:,i],0,15.0,20.)
    print(etiquetes[i],round(intercept[i],e_intercept[i]), round(slope[i],e_slope[i]))

#Gràfic amb l'ajust lineal tirat per a la gamma   
plt.figure()
y = ln_moms_corr.copy()
x = valors_q.copy()
for i in range(len(etiquetes)):
    plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
    plt.plot(x_fit[i], slope[i]*x_fit[i]+intercept[i], '--', color='black')
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,32,5))
plt.xlim(0,20)
plt.ylim(0,30)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \frac{\langle \tau^q \rangle}{\Gamma (q+1)}$')
#plt.savefig('moments_corr_plot_ajust_lineal.png', dpi=300, bbox_inches='tight')
plt.close()


#%%

#--- WEIBULL I Q-EXPONENTIAL DISTRIBUTIONS
def weibull_moments(x,a,b):
    y = x*np.log(a)+np.log(gamma(1+x/b))
    return y

popt = []
pcov = []
x = valors_q.copy()
x_mask = np.zeros(x.shape)
x_mask = (x <= 0.5) & (x>=0.0)
x_fit = x[x_mask]
for i in range (0, len(etiquetes)):
    y = ln_moms[:,i][x_mask]
    popt_i, pcov_i = curve_fit(weibull_moments, x_fit, y, maxfev=10000)
    popt.append(popt_i)
    pcov.append(pcov_i)

x = valors_q.copy()

for i in range(len(etiquetes)):
    y = ln_moms[:,i].copy()
    plt.plot(x, y, 'o', markersize=1, color=colors[i],label=etiquetes[i])
    plt.plot(x, weibull_moments(x,popt[i][0],popt[i][1]), '--', color='black')
plt.xticks(np.linspace(0,20,5))
plt.yticks(np.arange(0,70,10))
plt.xlim(0,20)
plt.ylim(0,70)
plt.legend(frameon=False)
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.xlabel('q')
plt.ylabel(r'$\ln \langle \tau^q \rangle$')
#plt.savefig('ajust_weibull_q_0a05.png', dpi=300, bbox_inches='tight')
plt.close()

# def weibull(x,a,b):
#     y = (b/a)*(x/a)**(b-1)*np.exp(-(x/a)**b)
#     return y

llista_distribucions = [dades[0][:,6]]
llista_i_distribucions, recorrer_distribucions = dis_decomulada(llista_distribucions)
#decomulada de la Weibull
valors_weibull = np.exp(-(recorrer_distribucions/(popt[0][0]))**popt[0][1])


print('\n Ajust Weibull, lambda, k i errors')
for i in range (0,len(etiquetes)):
    sigma = np.sqrt(np.diag(pcov[i]))
    print(etiquetes[i], popt[i][0],sigma[0],popt[i][1], sigma[1])



y = [llista_i_distribucions[0],valors_weibull]
x = recorrer_distribucions
etiquetes = ['Experimental', 'Weibull']
plt.figure()
datasets = len(y)
for i in range (0, datasets):
    y[i] = np.where(y[i] == 0, np.nan, y[i])
    plt.plot(x,y[i],'-',label=etiquetes[i])
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.ylim(0,None)
plt.xlim(0,None)
plt.xlabel(r'Temps (s)')
plt.ylabel(r'Probabilitat')
plt.legend(frameon=False)
#plt.savefig('decomulada_distribucions_q_0a05', dpi=300, bbox_inches='tight')
plt.close()


#Ara l'ajustarem directament sobre la decomulada

def deco_weibull(x,a,b):
    y = np.exp(-(x/a)**b)
    return y

x = recorrer_distribucions.copy()
y = llista_i_distribucions[0].copy()

for i in range (0, len(etiquetes)):
    popt_i, pcov_i = curve_fit(deco_weibull, x, y, maxfev=10000)
    popt.append(popt_i)
    pcov.append(pcov_i)
    
y = [llista_i_distribucions[0],deco_weibull(x, popt[0][0],popt[0][1])]
etiquetes = ['Experimental', 'Weibull']
plt.figure()
datasets = len(y)
for i in range (0, datasets):
    y[i] = np.where(y[i] == 0, np.nan, y[i])
    plt.plot(x,y[i],'-',label=etiquetes[i])
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.ylim(0,None)
plt.xlim(0,None)
plt.xlabel(r'Temps (s)')
plt.ylabel(r'Probabilitat')
plt.legend(frameon=False)
#plt.savefig('decomulada_distribucions_ajusatda_directament', dpi=300, bbox_inches='tight')
plt.close()

def decomulada_normal(x, mu, sigma):
    from scipy.stats import norm
    mu, sigma = 0, 1
    a, b = 0, 1
    area = np.zeros(x.shape)
    for i in range (0,len(x)):
        area[i] = norm.cdf((-x[i]-mu)/sigma, 0, 1) - norm.cdf((x[i]-mu)/sigma, 0, 1)
    return area

plt.figure()
plt.plot(recorrer_distribucions, llista_i_distribucions[0], label='Experimental')
plt.plot(recorrer_distribucions, 1-decomulada_normal(recorrer_distribucions,ln_moms[np.where(valors_q == 1.0),0][0], ln_moms[np.where(valors_q == 2.0),0][0]),label = 'Gaussiana')
plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
plt.ylim(0,None)
plt.xlim(0,None)
plt.xlabel(r'Temps (s)')
plt.ylabel(r'Probabilitat')
plt.legend(frameon=False)
#plt.savefig('decomulada_amb_gaussiana', dpi=300, bbox_inches='tight')
plt.close()
#%%
# --- JUSTOS MULTIFRACTALS, HEURÍSTICS I COMPROVACIONS

# #fem ajustos multifactals, mateix codi si canviem la funcio ens erveix
# def ajust_multifractal(x,A,b,c):
#     y = x*A + b*x**c
#     return y

# popt = []
# pcov = []
# for i in range (0, len(etiquetes)):
#     popt_i, pcov_i = curve_fit(ajust_multifractal, valors_q, ln_moms_corr[:,i], maxfev=10000)
#     popt.append(popt_i)
#     pcov.append(pcov_i)

# y = ln_moms_corr.copy()
# x = valors_q.copy()
# x_mask = np.zeros(x.shape)
# x_mask = (x <= 2.5) & (x>=0.5)
# x_fit = x[x_mask]

# for i in range(len(etiquetes)):
#     plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
#     plt.plot(x_fit, ajust_multifractal(x_fit,popt[i][0],popt[i][1],popt[i][2]), '--', color='black')
# # plt.xticks(np.arange(0,3.0,0.5))
# # plt.yticks(np.arange(0,10,2))
# # plt.xlim(0,12)
# # plt.ylim(0,10)
# plt.legend(frameon=False)
# plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
# plt.xlabel('q')
# plt.ylabel(r'$\ln \frac{\langle \tau^q \rangle}{\Gamma (q+1)}$')
# #plt.savefig('ajust_multifractal_gamma_alfa.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()


# popt = []
# pcov = []
# for i in range (0, len(etiquetes)):
#     popt_i, pcov_i = curve_fit(ajust_heuristic, valors_q, ln_moms_corr[:,i], maxfev=10000000)
#     popt.append(popt_i)
#     pcov.append(pcov_i)

# y = ln_moms_corr.copy()
# x = valors_q.copy()

# for i in range(len(etiquetes)):
#     plt.plot(x, y[:, i], 'o', markersize=1, color=colors[i],label=etiquetes[i])
#     plt.plot(x, ajust_heuristic(x,popt[i][0],popt[i][1],popt[i][2],popt[i][3]), '--', color='black')
# plt.xticks(np.linspace(0,20,5))
# plt.yticks(np.arange(0,30,5))
# plt.xlim(0,20)
# plt.ylim(0,70)
# plt.legend(frameon=False)
# plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
# plt.xlabel('q')
# plt.ylabel(r'$\ln \langle \tau^q \rangle$')
# #plt.savefig('moments_gamma_ajust_heuristic.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

# a = popt[:][0]
# b1 = popt[:][1]
# b = popt[:][2]
# c = popt[:][3]

# for i in range(0,len(etiquetes)):
#     sigma = np.sqrt(np.diag(pcov[i]))
#     print(etiquetes[i], popt[i][0],sigma[0],popt[i][1], sigma[1],popt[i][2], sigma[2],1/popt[i][3]+1,1/popt[i][3]**2*sigma[3] )
#     plt.plot(popt[i][1]*valors_q**popt[i][3],popt[i][1]/popt[i][2]*((1/valors_q)*ln_moms_corr[:,i]-popt[i][0]),'o', markersize=1, color=colors[i],label=etiquetes[i])
# # plt.xticks(np.arange(0,4,0.5))
# # plt.yticks(np.arange(0,1.1,0.2))
# # plt.xlim(0,4.0)
# # plt.ylim(0,1.0)
# plt.legend(frameon=False)
# plt.tick_params(axis='both', which='both', top=True, right=True, direction='out')
# plt.xlabel(r'$b_1 q^{c}$')
# plt.ylabel(r'$f_{H}$')
# #plt.savefig('ajust_heuristic_gamma_comprvacio.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

#%% Busquem 1/f noise
from scipy.signal import welch

data = dades[0][:,6]

# FFT directa
fft_vals = np.fft.fft(data)
fft_freqs = np.fft.fftfreq(len(data))

# PSD reial i positiva
psd = np.abs(fft_vals)**2

# Agafem només freqüències positives
mask = fft_freqs > 0
fft_freqs = fft_freqs[mask]
psd = psd[mask]

# Log-log
log_f = np.log10(fft_freqs)
log_psd = np.log10(psd)

plt.loglog(fft_freqs, psd, 'o')
# plt.savefig('1f.png')
