# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 23:27:06 2022

@author: Dirsa Feliciano / Orlando Arroyo / Daniela Novoa
"""
#%% Importar librerias
from openseespy.opensees import *
import numpy as np
import opsvis as opsv
import matplotlib.pyplot as plt
import analisis as an
import utilidades as ut
import multiprocessing
import time
import seaborn as sb
import pickle
# import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from joblib import Parallel, delayed 
#---------------------------------------------------------------------#
#%% Variables necesarias

records= ["GM01.txt", "GM02.txt", "GM03.txt", "GM04.txt", "GM05.txt", "GM06.txt", "GM07.txt", "GM08.txt", "GM09.txt", "GM10.txt", "GM11.txt", "GM12.txt", "GM13.txt", "GM14.txt", "GM15.txt",
         "GM16.txt", "GM17.txt", "GM18.txt", "GM19.txt", "GM20.txt", "GM21.txt", "GM22.txt", "GM23.txt", "GM24.txt", "GM25.txt", "GM26.txt", "GM27.txt", "GM28.txt", "GM29.txt", "GM30.txt",
          "GM31.txt", "GM32.txt", "GM33.txt", "GM34.txt", "GM35.txt", "GM36.txt", "GM37.txt", "GM38.txt", "GM39.txt", "GM40.txt", "GM41.txt", "GM42.txt", "GM43.txt", "GM44.txt"]
Nsteps= [3000, 3000, 2000, 2000, 5590, 5590, 4535, 4535, 9995, 9995, 7810, 7810, 4100, 4100, 4100, 4100, 5440, 5440, 6000, 6000, 2200, 2200, 11190, 11190, 7995, 7995, 7990, 7990, 2680, 2300, 8000, 8000, 2230, 2230, 1800, 1800, 18000, 18000, 18000, 18000, 2800, 2800, 7270, 7270]
DTs= [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.01,0.01,0.01,0.01,0.005,0.005,0.05,0.05,0.02,0.02,0.0025,0.0025,0.005,0.005,0.005,0.005,0.02,0.02,0.005,0.005,0.01,0.01,0.02,0.02,0.005,0.005,0.005,0.005,0.01,0.01,0.005,0.005]
SpectrumFactor = [0.484640765,0.338421756,0.386628219,0.308779724,0.175089881,0.312791906,0.80124682,0.405707818,0.643528131,0.5168247,0.423242076,0.326193659,0.304726603,0.415798489,0.832332932,0.8744116,0.507859107,0.393417493,1.673192442,2.335347059,0.968938985,0.911432171,0.368427384,0.306190804,0.230296494,0.451130491,0.374317312,0.36178636,0.342996872,0.323438672,0.612739273,1.063474848,0.600833795,0.49549825,0.43118056,0.29284453,0.672893134,0.501345323,0.341215077,0.518470373,0.838847295,1.219406411,0.52935428,0.500858943]

index2 = range(len(records))

# Elementos y nodos a grabar información
node_record = [30,31,32]
ele_record = [12]

n = 150 #Número de modelos
mfc = 21000
cv = 0.25
sfc = cv*mfc
Fc = np.random.normal(mfc,sfc,n)
mfy = 441900
cv2 = 0.1
sfy = cv2*mfy 
Fy = np.random.normal(mfy,sfy,n)
mfm = 1.01*1000
cv3 = 0.35
sfm = cv3*mfm 
Fm = np.random.normal(mfm,sfm,n)
    
variables=[]
for i in range(n):
    variables.append([Fc[i],Fy[i],Fm[i]])

#%% Funciones de generación
# Definir el modelo

def runDyn(ind,var):
    #%% Definir el modelo - Función
    wipe()
    
        #
    model('basic','-ndm',2,'-ndf',3)
        
    #%% Geometría del modelo
    H=2.7
    L= 3.0
    
    xloc=[0.0,L,2*L]
    yloc=[0.0,H,2*H]
    
    ny = len(yloc) # número de nodos que habrá en el sentido Y
    nx = len(xloc) # número de nodos que habrá en el sentido X
    
    for i in range(nx):
        for j in range(ny):
            nnode = 10*(i+1)+j # genera el tag del nodo, iniciando en 1000,1001... para el muro más a la izquierda, en dirección hacia arriba. luego sigue 2000,2001, luego 3000,3001... y así
            node(nnode,xloc[i],yloc[j]) # genera el nodo
    
    # plt.figure()
    # opsv.plot_model()
    
    print('Nodos generados')
    muros=[1,1]
    vigas=[1,1]
    
    # apoyos
    empotrado = [1,1,1]
    grado2 = [1,1,0]
    
    # para colocarlos todos al nivel 0.0
    fixY(0.0,*empotrado)
    
    print('Restricciones asignadas')
    # %% ASIGNACIÓN DE DIAFRAGMAS
    # =========================================
    diafragma=1
    if diafragma == 1:
        for j in range(1,ny):
            for i in range(1,nx):
                masternode = 10 + j # nodo maestro del piso
                slavenode = 10*(i+1) + j # cada uno de los nodos esclavos
                print(masternode,slavenode)
                equalDOF(masternode,slavenode,1) # asigna igualdad para la dirección X
        print('Diafragmas asignados')
    
    # plt.figure()
    # opsv.plot_model()
    
    #%% Materiales
    
    #------Concreto sin confinar--------
    UnConf= 3
    UnConfminmax= 4
    pint=5
    fc= var[0]
    E= 4400*np.sqrt(fc/1000)*1000
    ec= 2*fc/E
    fcu= 0.2*fc
    # ecu= 0.01*2.5
    lam=0.5
    ft=200
    Ets= 80
    Gfc= 25
    Lcol = H*1000
    e20 = ut.e20Lobatto2(Gfc, Lcol, pint, fc/1000, E/1000, ec)
    uniaxialMaterial('Concrete02', UnConf, -fc, -ec, -fcu, -e20,lam,ft,Ets)
    # uniaxialMaterial('MinMax', UnConfminmax, UnConf, '-min', -e20, '-max', 0.0)
    
    #------Concreto confinado--------
    Conf= 1
    Confminmax= 2
    fcc= fc*1.3
    ecc= 2*fcc/E
    fccu= 0.2*fcc
    # eccu= 5*ecc*2.5
    lam= 0.5
    ft= 200
    Ets= 80
    Gfcc= Gfc*6
    e20cc = ut.e20Lobatto2(Gfcc, Lcol, pint, fcc/1000, E/1000, ecc)
    uniaxialMaterial('Concrete02', Conf, -fcc, -ecc, -fccu, -e20cc,lam,ft,Ets)
    # uniaxialMaterial('MinMax', Confminmax, Conf, '-min', -e20cc, '-max', 0.0)
    
    #------acero------------------------
    Steel= 5
    Steelminmax= 6
    Fy= var[1]
    Fu= 632.90*1000
    esh= 0.0188
    eult= 0.1034
    uniaxialMaterial('Hysteretic',Steel, Fy, 0.002, Fu, eult, 0.2*Fu, 0.15, -Fy, -0.002, -Fu, -eult, -0.2*Fu, -0.15, 1, 1, 0.0, 0.0)
    uniaxialMaterial('MinMax', Steelminmax, Steel, '-min', -0.015, '-max', 0.05)
    #--------------Mampostería-----------
    masonry= 7
    masonryminmax= 8
    
    fm= var[2]
    Em= 900*fm
    em= 2*fm/Em
    fmu= 0.36*1000
    emu= em*3
    uniaxialMaterial('Concrete01',masonry,-fm, -em, -fmu, -emu)
    uniaxialMaterial('MinMax', masonryminmax, masonry, '-min', -0.006, '-max', 0.0)
    # -------------Transformaciones---------
    lineal = 1
    geomTransf('Linear',lineal)
    
    pdelta = 2
    geomTransf('PDelta',pdelta)
    
    cor = 3
    geomTransf('Corotational',cor)
    
    
    #%% ----------Secciones y elementos--------------------
    
    # -----------Sección columna---------
    Bcol = 0.20
    Hcol = 0.25
    c = 0.05  # recubrimiento 
    
    # creación de la sección de fibra
    y1col = Hcol/2.0
    z1col = Bcol/2.0
    
    y2col = 0.5*(Hcol-2*c)/3.0
    
    nFibZ = 1
    nFibZcore= 10
    nFib = 20
    nFibCover, nFibCore = 3, 16
    As4 = 0.000127
    
    sec20x25 = 1
    
    col20x25 = [['section', 'Fiber', sec20x25, '-GJ', 1.0e6],
                 ['patch', 'rect', UnConf, nFibCore, nFibZcore, c-y1col, c-z1col, y1col-c, z1col-c],
                 ['patch', 'rect', UnConf, nFib, nFibZ, -y1col, -z1col, y1col, c-z1col],
                 ['patch', 'rect', UnConf, nFib, nFibZ, -y1col, z1col-c, y1col, z1col],
                 ['patch', 'rect', UnConf, nFibCover, nFibZ, -y1col, c-z1col, c-y1col, z1col-c],
                 ['patch', 'rect', UnConf, nFibCover, nFibZ, y1col-c, c-z1col, y1col, z1col-c],
                 ['layer', 'straight', Steelminmax, 2, As4, y1col-c, z1col-c, y1col-c, c-z1col],
                 ['layer', 'straight', Steelminmax, 2, As4, c-y1col, z1col-c, c-y1col, c-z1col]]
    matcolor = ['lightgrey', 'r', 'w','w']
    # opsv.plot_fiber_section(col20x25, matcolor=matcolor)
    # plt.axis('equal')
    opsv.fib_sec_list_to_cmds(col20x25)
    
    #------------------Puntales--------------------------------
    
    tinf=0.10
    hcol= H
    lcol= L-Hcol/2 
    hinf= H-Hcol/2 
    linf= L-Hcol
    rinf=  pow((pow(hinf,2.0)+pow(linf,2.0)),0.5)
    Ldiag= pow((pow(hcol,2.0)+pow(lcol,2.0)),0.5)
    a= rinf/4.0; #Pauly and Prisley
    A= (a*tinf*Ldiag)/rinf
    
    
    #%% Para elementos 
    pint = 5
    beamIntegration('Lobatto', sec20x25, sec20x25,pint)
    for i in range(nx):
        col = []
        for j in range(ny-1):
            nodeI = 10*(i+1) + j # nodo inicial de la columna
            nodeJ = 10*(i+1) + (j+1) # nodo final de la columna
            eltag = 12*(i+1) + j # tag de la columna
            col.append(eltag) # acumula el tag
            element('forceBeamColumn',eltag, nodeI,nodeJ,pdelta,sec20x25)
     
    print('Columnas')
    # plt.figure()        
    # opsv.plot_model()
    
    
    tagvigas=[]
    for j in range(1,ny):
        for i in range(nx-1):
            if vigas[i] == 1:
                nodeI = 10*(i+1) + j # nodo inicial de la viga
                nodeJ = 10*(i+2) + j# nodo final de la viga
                eltag = 101*(i+1) + j # tag de la viga
                tagvigas.append(eltag) # guarda los tags de las vigas
                element('forceBeamColumn',eltag, nodeI,nodeJ,lineal,sec20x25)
    print('vigas generadas')
    # plt.figure()        
    # opsv.plot_model()
    
    
    tagmuros1=[]
    tagmuros2=[]
    for j in range(1,ny):
        for i in range(nx-1):
            if muros[i] == 1:
                nodeI = 10*(i+1) + j # nodo inicial de la viga
                nodeJ = 10*(i+2) + j-1# nodo final de la viga
                nodeI2= 10*(i+1) + j-1
                nodeJ2= 10*(i+2) +j
                eltag1 = 1000*(i+1) + j # tag de la viga
                tagmuros1.append(eltag1) # guarda los tags de las vigas
                eltag2= 10000*(i+1) + j 
                tagmuros2.append(eltag2)
                element('Truss',eltag1, nodeI,nodeJ ,A,masonry)
                element('Truss',eltag2, nodeI2,nodeJ2 ,A,masonry)
                
    print('muros generados')
    
    
    # plt.figure()        
    # opsv.plot_model()
    # %% Cargas y masas
    #  pattern('Plain', patternTag, tsTag, '-fact', fact)
    wcol=Bcol*Hcol*H*24
    timeSeries('Linear', 1)
    pattern('Plain',1,1)
    
    for i in range(nx):
        for j in range(ny):
            nnode2 = 10*(i+1)+j # genera el tag del nodo, iniciando en 1000,1001... para el muro más a la izquierda, en dirección hacia arriba. luego sigue 2000,2001, luego 3000,3001... y así
            # mass(nnode2,masas[i][j],masas[i][j],0.0) # genera el nodo
            load(nnode2,0.0,-1*wcol,0.0)
    
    mass(11,3.53,3.53, 0.0)
    mass(21,4.28,4.28, 0.0)
    mass(31,3.53,3.53, 0.0)
    mass(12,1.37,1.37, 0.0)
    mass(22,1.66,1.66, 0.0)
    mass(32,1.37,1.37, 0.0)  
      
    eleLoad('-ele',102,'-type','-beamUniform',-14)
    eleLoad('-ele',203,'-type','-beamUniform',-14)
    eleLoad('-ele',103,'-type','-beamUniform',-2)
    eleLoad('-ele',204,'-type','-beamUniform',-2)
    
    #%% Análisis
    # plt.figure()
    # concargas = opsv.plot_supports_and_loads_2d()
    
    # eig = eigen(1)
    # T1 = 2*3.1416/np.sqrt(eig[0])
    # opsv.plot_mode_shape(1)
    
    an.gravedad()
    loadConst('-time',0.0)
    # reactions()
    # n1 = nodeReaction(10)
    # n2 = nodeReaction(20)
    # n3 = nodeReaction(30)
    # defo = opsv.plot_defo(sfac = 500)
    
    #-------Pushover--------------------
    
    # columns=[12,24,36]
    # struts=[1001,10001,2001,20001]
    # todo = [12,24,36]
    # recorder('Element','-file','struts.out','-time','-ele',*struts,'globalForce')
    # # recorder('Element','-file','columns.out','-time','-ele',*columns,'globalForce')
    # recorder('Node','-file','node31.out','-time','-node',31,'-dof',1,'disp')
    # recorder('Node','-file','node61.out','-time','-node',61,'-dof',1,'disp')
    # recorder('Node','-file','node91.out','-time','-node',91,'-dof',1,'disp')
    
    # recorder('Node','-file','node32.out','-time','-node',32,'-dof',1,'disp')
    # recorder('Node','-file','node62.out','-time','-node',62,'-dof',1,'disp')
    # recorder('Node','-file','node92.out','-time','-node',92,'-dof',1,'disp')
      
    # dtecho2, Vbasal2, Fcol = an.pushover2(0.03*2*H, 0.001, nnode, 1, todo)
    tiempo,techo,Eds,node_disp,node_vel,node_acel,drift = an.dinamicoIDA4P(records[ind], DTs[ind], Nsteps[ind]+100, 0.04, 9.81*SpectrumFactor[ind]*1.5, 0.025, 32, 1, ele_record, node_record)
    
    return tiempo,techo,Eds,node_disp,node_vel,node_acel,drift, fc, Fy, ind

#%% Procesar pushover con todos los nucleos del Pc y en paralelo
num_cores = multiprocessing.cpu_count() # num nucleos en el PC
stime = time.time()

resultados = Parallel(n_jobs=num_cores)(delayed(runDyn)(ii, ff) for ii in index2 for ff in variables) # loop paralelo

etime = time.time()

ttotal = etime - stime

print('tiempo de ejecucion: ',ttotal,'segundos')


#%% Grabar resultados
filename = '150sim2_1.5'
outfile = open(filename,'wb')
pickle.dump(resultados,outfile)
outfile.close()

#%% Cargar resultados

# filename = '150sim2_1.5'
# infile = open(filename,'rb')
# resultados = pickle.load(infile)
# infile.close()

#%% Graficar

techomax = np.zeros(n*len(index2))
residual = np.zeros(n*len(index2))
drift_s1 = np.zeros(n*len(index2))
hbuild = 9

# este loop procesa todos los resultados
for index,result in enumerate(resultados):
    dtecho = result[1]
    drifts1 = result[6]
    techomax[index] = np.max(np.abs(dtecho))/hbuild*100
    residual[index] = np.abs(dtecho[-1])/hbuild*100
    drift_s1[index] = np.max(np.abs(drifts1[:,0]))*100
    
# Aquí se organizan los resultados. Cada columna es un sismo y en las filas van las simulaciones

nEQ = len(index2)
techomax2 = techomax.reshape(nEQ,n).transpose()
residual2 = residual.reshape(nEQ,n).transpose()
drift_s1 = drift_s1.reshape(nEQ,n).transpose()

# df_techo = pd.DataFrame(data = techomax2[:,1:3])
eqind = 2

sb.boxplot(data = drift_s1[:,eqind])
plt.ylabel('Max first story drift (%)')
# plt.ylim([1.3,1.7])
plt.show()

# sb.ecdfplot(data = drift_s1[:,eqind])
# plt.xlabel('Max drift 1st story (%)')
# plt.ylabel('Cumulative Probability')
# plt.show()

# %% obtención de las ECDF

# Ejemplo:
ecdf = ECDF(drift_s1[:,eqind]) # calcula la distribución acumulada empírica
plt.plot(ecdf.x,ecdf.y) # calcula la distribución acumulada empírica los .x y .y son respectivamente X y Y
plt.show()

val = np.interp(0.7,ecdf.x,ecdf.y) # Así se interpola en un valor

CDF_drifts1 = [] # para almacenar las CDf de las derivas del primer piso
CDF_techo = []
for i in range(nEQ):
    ecdf1 = ECDF(drift_s1[:,i])
    ecdf2 = ECDF(techomax2[:,i])
    CDF_drifts1.append(ecdf1)
    CDF_techo.append(ecdf2)
    
#%% Cálculo de probabilidades

lim1 = 1.0 # 0.5% de daño para moderado (por ejemplo)

probabilities = []

for ecdf in CDF_drifts1:
    prob = 1-np.interp(lim1,ecdf.x,ecdf.y)
    probabilities.append(prob)
    

ecdf3 = CDF_drifts1[3]
plt.plot(ecdf3.x,ecdf3.y) # calcula la distribución acumulada empírica los .x y .y son respectivamente X y Y
plt.show()

ecdf5 = CDF_drifts1[9]
plt.plot(ecdf5.x,ecdf5.y) # calcula la distribución acumulada empírica los .x y .y son respectivamente X y Y
plt.show()

#%%
sb.boxplot(data = probabilities)
# plt.ylim([1.3,1.7])
plt.show()

