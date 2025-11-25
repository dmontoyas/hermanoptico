import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium # La biblioteca para entornos de RL
from gymnasium import spaces
import torch as th
from stable_baselines3 import PPO # Importamos el agente PPO estándar

# Importaciones del Framework
from dreamongymv2.simNetPy import * # Importa el núcleo del simulador (Flex Net Sim), clases como Controller, BitRate, Connection, Network.
from dreamongymv2.gym_basic import * # Importa la capa de adaptación de Gymnasium
from dreamongymv2.gym_basic.envs import RlOnEnv # Importa la clase principal del Entorno de RL

# Parche de compatibilidad
sys.modules["gym"] = gymnasium

import time
start_time = time.perf_counter()

def get_reward():
    # Consultamos al simulador si la última acción tuvo éxito
    status = env.unwrapped.getSimulator().lastConnectionIsAllocated()
    
    if status.name == Controller.Status.Not_Allocated.name:
        value = -1  # Conexión bloqueada
    else:
        value = 1 # Conexión aceptada

    return value

def get_state():
    """
    Devolveremos un vector plano que concatena la ocupación de las K rutas.
    Tamaño del vector: K_PATHS * TOTAL_SLOTS
    Valor 0 = Libre, 1 = Ocupado
    """
    sim = env.unwrapped.getSimulator()

    # Obtenemos la solicitud actual
    event = sim.connectionEvent
    src = event.source
    dst = event.destination

    # Obtenemos el índice del bitrate del evento (el entero 0, 1, 2...)
    bitrate_idx = event.bitRate
    # Accedemos a la lista maestra de BitRates del simulador
    bitrate_obj = sim.bitRates[bitrate_idx]
    # Preguntamos al objeto cuántos slots requiere
    req_slots = bitrate_obj.getNumberofSlots(0)
    max_slots = sim.bitRates[4].slots[0]

    # Normalizamos diviendo por el máximo número de slots posibles
    norm_req_slots = req_slots / max_slots

    # Obtenemos las K rutas posibles para este par origen-destino
    k_paths_list = sim.controller.path[src][dst]
    
    # Vector de observación
    obs = []

    # Lista para guardar el % de ocupación de cada ruta
    routes_utilization = []

    # Iteramos solo sobre las primeras K rutas (acciones posibles del agente)
    for k in range(K_PATHS):
        
        # Estado de la ruta k
        path_occupancy = [0] * TOTAL_SLOTS
        
        # Si la ruta existe (por si hay pares con menos de K rutas)
        if k < len(k_paths_list):
            route_links = k_paths_list[k]
            
            # Construimos la visión "End-to-End" de la ruta (lógica OR)
            # Igual que en tu heurística, pero solo para mirar, no para asignar
            for link_obj in route_links:
                link = sim.controller.network.getLink(link_obj.id)
                for slot in range(TOTAL_SLOTS):
                    # Si el slot está ocupado en este enlace, marcamos la ruta como ocupada en ese slot
                    if link.getSlot(slot): 
                        path_occupancy[slot] = 1
        else:
            # Si no existe la ruta k, la marcamos como totalmente ocupada (bloqueada)
            # para que el agente aprenda a no elegirla.
            path_occupancy = [1] * TOTAL_SLOTS
            
        obs.extend(path_occupancy)

        utilization = sum(path_occupancy) / float(TOTAL_SLOTS) # % de Ocupación
        routes_utilization.append(utilization)

    # Añadir la información de la solicitud al final
    # Construcción Final del Vector
    # [Mapa de bits (500)] + [Utilización por ruta (5)] + [Req Slots (1)]
    obs.extend(routes_utilization)
    obs.append(norm_req_slots)

    return np.array(obs, dtype=np.float32)

def rl_agent_allocator(src: int, dst: int, b: BitRate, c: Connection, n: Network, path, action):
    """
    'action' es un número entero (0, 1, ... K-1) decidido por PPO.
    """
    actionSpace = len(path[src][dst])

    if action is not None:
        if action == actionSpace:
            action = action - 1
    else:
        action = 0

    # Por si el agente alucina un índice fuera de rango
    if action > actionSpace:
        return Controller.Status.Not_Allocated, c
    
    # Acción: el agente selecciona una ruta
    selected_route = path[src][dst][action]
    
    # Se aplica First-Fit solo en esa ruta
    numberOfSlots = b.getNumberofSlots(0)
    
    total_slots = n.getLink(0).getSlots()
    general_link = [False] * total_slots
    
    # Construimos máscara de ocupación de la ruta seleccionada
    for link_obj in selected_route:
        link = n.getLink(link_obj.id)
        for slot in range(link.getSlots()):
            general_link[slot] = general_link[slot] or link.getSlot(slot)
            
    # Buscar hueco
    currentNumberSlots = 0
    currentSlotIndex = 0
    for j in range(len(general_link)):
        if not general_link[j]:
            currentNumberSlots += 1
        else:
            currentNumberSlots = 0
            currentSlotIndex = j + 1
        
        if currentNumberSlots == numberOfSlots:
            # Asignar
            for k_link_obj in selected_route:
                c.addLink(k_link_obj, fromSlot=currentSlotIndex, toSlot=currentSlotIndex+numberOfSlots)
            return Controller.Status.Allocated, c
            
    # Si First-Fit falla en la ruta elegida por el agente, es un bloqueo.
    return Controller.Status.Not_Allocated, c


K_PATHS = 5       # El agente elegirá entre las 5 mejores rutas
TOTAL_SLOTS = 100 # 100 slots por enlace
input_size = (K_PATHS * TOTAL_SLOTS) + K_PATHS + 1

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

print("Simulación Agentes")

# Cargamos agente entrenado
model = PPO.load("hermanoptico_v4")

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
        # Acción
        env.action_space = spaces.Discrete(K_PATHS)
        # Observación
        env.observation_space = spaces.Box(low=0, high=1, shape=(input_size,), dtype=np.float32)
        
        obs, info = env.reset()

        env.setRewardFunc(get_reward)
        env.setStateFunc(get_state)
        # Cargamos topología
        env.initEnviroment(fileDirectory + "/Biobio_Net.json", 
                   fileDirectory + "/Biobio_Net_routes.json", 
                   fileDirectory + "/Biobio_Net_bitrate.json")

        # Configuramos el simulador
        env.getSimulator().setGoalConnections(TOTAL_CONNECTIONS)
        env.getSimulator().setLambda(TRAFFIC_LAMBDA) # Arrival rate of new connection requests
        env.getSimulator().setMu(TRAFFIC_MU) # Departure rate of active connections.
        env.getSimulator().setAllocator(rl_agent_allocator)
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

        # A veces env.reset() devuelve un escalar (shape ()) porque no sincroniza
        # bien con el simulador ya iniciado. Si pasa eso, forzamos la lectura del estado real.
        if np.shape(obs) == ():
            obs = get_state()
        
        # Bucle de simulación (correr las 'total_conn' solicitudes)
        for _ in range(TOTAL_CONNECTIONS):
            # Usamos el modelo entrenado para predecir
            action, _states = model.predict(obs, deterministic=True)
            # Paso del entorno
            obs, reward, terminated, truncated, info = env.step(action)

            # Si por alguna razón env.step devuelve un escalar
            if np.shape(obs) == ():
                obs = get_state()
            
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

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nTiempo de simulación: {elapsed_time/60.} minutos")

print("\n--- EXPERIMENTO COMPLETO ---")
print("Resumen de resultados (Total Conexiones: {media, std}):")
print(final_results)
