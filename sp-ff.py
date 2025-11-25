import os
import sys
import numpy as np
import gymnasium # La biblioteca para entornos de RL

# Importaciones del Framework
from dreamongymv2.simNetPy import * # Importa el núcleo del simulador (Flex Net Sim), clases como Controller, BitRate, Connection, Network.
from dreamongymv2.gym_basic import * # Importa la capa de adaptación de Gymnasium
from dreamongymv2.gym_basic.envs import RlOnEnv # Importa la clase principal del Entorno de RL

# Parche de compatibilidad.
sys.modules["gym"] = gymnasium  

#####################
# Funciones Dummy
# El entorno 'env.step()' las necesita para funcionar, aunque no usemos su resultado.
def dummy_reward():
    return 0

def dummy_state():
    # Debe devolver un estado válido, como una lista.
    return [0]
#####################

#####################
# Asignador Heurístico SP-FF
def sp_ff_allocator(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    """
    Implementación la heurística Shortest Path + First-Fit (KSP-FF).
    
    1. Selecciona la ruta más corta.
    2. Intenta asignar espectro usando First-Fit.
    3. Si tiene éxito, asigna y retorna.
    
    El parámetro 'action' se ignora por completo.
    """

    # 'b' es un objeto BitRate. b.getNumberofSlots(0) nos da los slots solicitados.
    # print(f"--> Solicitud Evento: (Fuente: {src}, Destino: {dst}, Slots Req: {b.getNumberofSlots(0)})")

    numberOfSlots = b.getNumberofSlots(0) 

    # Obtenemos los links de la ruta más corta
    shortest_path = path[src][dst][0]

    # First-Fit
    # Creamos 'general_link' para esta ruta
    total_slots = n.getLink(0).getSlots()
    general_link = [False] * total_slots

    # Fusionamoa la ocupación de todos los enlaces de esta ruta.
    # Acá estamos implementando la restricción de continuidad de espectro en los enlaces
    for link_obj in shortest_path:
        link = n.getLink(link_obj.id) # Se obtiene el objeto Link completo
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(slot)

    # Busca el primer hueco (First-Fit) en 'general_link'
    # Acá estamos implementando la restricción de contigüidad
    currentNumberSlots = 0
    currentSlotIndex = 0
    for j in range(len(general_link)):
        if not general_link[j]: # Si el slot está libre
            currentNumberSlots += 1
        else: # Si está ocupado, reiniciar conteo
            currentNumberSlots = 0
            currentSlotIndex = j + 1
            
        # Si encontramos un hueco lo suficientemente grande
        if currentNumberSlots == numberOfSlots:
            # Asignamos estos slots
            for k_link_obj in shortest_path:
                c.addLink(
                    k_link_obj, fromSlot=currentSlotIndex, toSlot=currentSlotIndex+currentNumberSlots)
                
            return Controller.Status.Allocated, c

    # Bloqueo: Si el bucle 'for' termina.
    return Controller.Status.Not_Allocated, c
#####################

#####################
# Inicio del Script Principal
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

TOTAL_CONNECTIONS = 25000
N_SIMULATIONS = 10

# Parámetros de tráfico (Erlang = Lambda / Mu)
list_traffic_lambda = np.arange(20,550,50)
TRAFFIC_MU = 1

# Diccionario para guardar los resultados finales
final_results = {}

print("Simulación heurística SP-FF")

# Itera sobre cada Erlang
for TRAFFIC_LAMBDA in list_traffic_lambda:
    
    print(f"\nProcesando Erlang: {TRAFFIC_LAMBDA} ")
    
    # Lista para guardar los BP de las N_SIMULATIONS
    list_bp_simulations = []

    # BUCLE INTERNO: Corre la simulación N_SIMULATIONS veces con semillas diferentes
    for i in range(N_SIMULATIONS):
        print(f"  ... Simulación {i+1}/{N_SIMULATIONS} (Semilla: {i})", end="")
        
        # Creamos un entorno nuevo para esta simulación
        env = RlOnEnv()
        obs, info = env.reset()
        env.setRewardFunc(dummy_reward)
        env.setStateFunc(dummy_state)

        # Cargamos topología
        env.initEnviroment(fileDirectory + "/Biobio_Net.json", fileDirectory + "/Biobio_Net_routes.json", 
                           fileDirectory + "/Biobio_Net_bitrate.json")

        # Configuramos el simulador
        env.getSimulator().setGoalConnections(TOTAL_CONNECTIONS)
        env.getSimulator().setLambda(TRAFFIC_LAMBDA) # Arrival rate of new connection requests
        env.getSimulator().setMu(TRAFFIC_MU) # Departure rate of active connections.
        env.getSimulator().setAllocator(sp_ff_allocator)
        env.getSimulator().setConfidence(0.95)

        # Fijamos semillas
        env.getSimulator().setSeedArrive(i)
        env.getSimulator().setSeedBitRate(i)
        env.getSimulator().setSeedDeparture(i)
        env.getSimulator().setSeedDst(i)
        env.getSimulator().setSeedSrc(i)
        
        # Inicializamos y empezamos
        env.getSimulator().init() # The simulator is prepared for execution
        env.start(verbose=False) # the simulation process starts
        
        # Bucle de simulación (correr las 'TOTAL_CONNECTIONS' solicitudes)
        for _ in range(TOTAL_CONNECTIONS):
            # La acción '0' es dummy, sp_ff_allocator la ignora
            obs, rewards, terminated, truncated, info = env.step(0) 
            
        # Obtenemos y guardamos el resultado 
        bp_simulation = env.getSimulator().getBlockingProbability()
        list_bp_simulations.append(bp_simulation)
        
        print(f" -> BP: {bp_simulation:.2e}")
        
        # Limpiamos el entorno 
        del env

    # Fin Bucle Interno

    # Calculamos media y sesviación estándar para este punto
    mean_bp = np.mean(list_bp_simulations)
    std_bp = np.std(list_bp_simulations)
    
    print(f"Resultados para Erlang {TRAFFIC_LAMBDA}:")
    print(f"  -> Media BP: {mean_bp:.2e}")
    print(f"  -> Desv. Estándar BP: {std_bp:.2e}")
    
    # Guardamos en el diccionario
    final_results[TRAFFIC_LAMBDA] = {'media': mean_bp, 'std': std_bp}

# Fin Bucle Externo

print("\n--- EXPERIMENTO COMPLETO ---")
print("Resumen de resultados (Total Conexiones: {media, std}):")
print(final_results)

