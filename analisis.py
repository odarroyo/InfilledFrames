# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:59:44 2022

@author: Orlando
"""
from openseespy.opensees import *
import matplotlib.pyplot as plt
import numpy as np

# ANALISIS DE GRAVEDAD
# =============================
def gravedad():
    
# Create the system of equation, a sparse solver with partial pivoting
    system('BandGeneral')

# Create the constraint handler, the transformation method
    constraints('Plain')

# Create the DOF numberer, the reverse Cuthill-McKee algorithm
    numberer('RCM')

# Create the convergence test, the norm of the residual with a tolerance of
# 1e-12 and a max number of iterations of 10
    test('NormDispIncr', 1.0e-12, 10, 3)

# Create the solution algorithm, a Newton-Raphson algorithm
    algorithm('Newton')

# Create the integration scheme, the LoadControl scheme using steps of 0.1
    integrator('LoadControl', 0.1)

# Create the analysis object
    analysis('Static')

    ok = analyze(10)
    
    if ok != 0:
        print('Análisis de gravedad fallido')
        sys.exit()
    else:
        print('Análisis de gravedad completado')
        



def dinamicoIDA4P(recordName,dtrec,nPts,dtan,fact,damp,IDctrlNode,IDctrlDOF,elements,nodes_control,modes = [0,2],Kswitch = 1,Tol=1e-8):
    
    # PARA SER UTILIZADO PARA CORRER EN PARALELO LOS SISMOS Y EXTRAYENDO LAS FUERZAS DE LOS ELEMENTOS INDICADOS EN ELEMENTS
    
    # record es el nombre del registro, incluyendo extensión. P.ej. GM01.txt
    # dtrec es el dt del registro
    # nPts es el número de puntos del análisis
    # dtan es el dt del análisis
    # fact es el factor escalar del registro
    # damp es el porcentaje de amortiguamiento (EN DECIMAL. p.ej: 0.03 para 3%)
    # IDcrtlNode es el nodo de control para grabar desplazamientos
    # IDctrlDOF es el grado de libertad de control
    # elements son los elementos de los que se va a grabar información
    # nodes_control son los nodos donde se va a grabar las respuestas
    # Kswitch recibe: 1: matriz inicial, 2: matriz actual
    
    maxNumIter = 10
    
    # creación del pattern
    
    timeSeries('Path',1000,'-filePath',recordName,'-dt',dtrec,'-factor',fact)
    pattern('UniformExcitation',  1000,   1,  '-accel', 1000)
    
    # damping
    nmodes = max(modes)+1
    eigval = eigen(nmodes)
    
    eig1 = eigval[modes[0]]
    eig2 = eigval[modes[1]]
    
    w1 = eig1**0.5
    w2 = eig2**0.5
    
    beta = 2.0*damp/(w1 + w2)
    alfa = 2.0*damp*w1*w2/(w1 + w2)
    
    if Kswitch == 1:
        rayleigh(alfa, 0.0, beta, 0.0)
    else:
        rayleigh(alfa, beta, 0.0, 0.0)
    
    # configuración básica del análisis
    wipeAnalysis()
    constraints('Plain')
    numberer('RCM')
    system('BandGeneral')
    test('EnergyIncr', Tol, maxNumIter)
    algorithm('Newton')    
    integrator('Newmark', 0.5, 0.25)
    analysis('Transient')
    
    # Otras opciones de análisis    
    tests = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
    algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

    # rutina del análisis
    
    Nsteps =  int(dtrec*nPts/dtan)
    dtecho = [nodeDisp(IDctrlNode,IDctrlDOF)]
    t = [getTime()]
    nels = len(elements)
    nnodos = len(nodes_control)
    Eds = np.zeros((nels, Nsteps+1, 6)) # para grabar las fuerzas de los elementos
    
    
    
    node_disp = np.zeros((Nsteps + 1, nnodos)) # para grabar los desplazamientos de los nodos
    node_vel = np.zeros((Nsteps + 1, nnodos)) # para grabar los desplazamientos de los nodos
    node_acel = np.zeros((Nsteps + 1, nnodos)) # para grabar los desplazamientos de los nodos
    drift = np.zeros((Nsteps + 1, nnodos - 1)) # para grabar la deriva de entrepiso
    
    for k in range(Nsteps):
        ok = analyze(1,dtan)
        # ok2 = ok;
        # En caso de no converger en un paso entra al condicional que sigue
        if ok != 0:
            print('configuración por defecto no converge en tiempo: ',getTime())
            for j in algoritmo:
                if j < 4:
                    algorithm(algoritmo[j], '-initial')
    
                else:
                    algorithm(algoritmo[j])
                
                # el test se hace 50 veces más
                test('EnergyIncr', Tol, maxNumIter*50)
                ok = analyze(1,dtan)
                if ok == 0:
                    # si converge vuelve a las opciones iniciales de análisi
                    test('EnergyIncr', Tol, maxNumIter)
                    algorithm('Newton')
                    break
                    
        if ok != 0:
            print('Análisis dinámico fallido')
            print('Desplazamiento alcanzado: ',nodeDisp(IDctrlNode,IDctrlDOF),'m')
            break
        
        for node_i, node_tag in enumerate(nodes_control):
            
            node_disp[k+1,node_i] = nodeDisp(node_tag,1)
            node_vel[k+1,node_i] = nodeVel(node_tag,1)
            node_acel[k+1,node_i] = nodeAccel(node_tag,1)
            if node_i != 0:
                drift[k+1,node_i-1] = (nodeDisp(node_tag,1) - nodeDisp(nodes_control[node_i-1],1))/(nodeCoord(node_tag,2) - nodeCoord(nodes_control[node_i-1],2))
                       

        for el_i, ele_tag in enumerate(elements):
                      
            Eds[el_i , k+1, :] = [eleResponse(ele_tag,'globalForce')[0],
                                 eleResponse(ele_tag,'globalForce')[1],
                                 eleResponse(ele_tag,'globalForce')[2],
                                 eleResponse(ele_tag,'globalForce')[3],
                                 eleResponse(ele_tag,'globalForce')[4],
                                 eleResponse(ele_tag,'globalForce')[5]]
            
        dtecho.append(nodeDisp(IDctrlNode,IDctrlDOF))
        t.append(getTime())
        
    # plt.figure()
    # plt.plot(t,dtecho)
    # plt.xlabel('tiempo (s)')
    # plt.ylabel('desplazamiento (m)')
    
    techo = np.array(dtecho)
    tiempo = np.array(t)
    wipe()
    return tiempo,techo,Eds,node_disp,node_vel,node_acel,drift
