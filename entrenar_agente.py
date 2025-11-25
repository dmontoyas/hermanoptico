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

# Parámetros globales
K_PATHS = 5       # El agente elegirá entre las 5 mejores rutas
TOTAL_SLOTS = 100 # 100 slots por enlace
TOTAL_CONNECTIONS = 25000

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


class ForceLongEpisodeWrapper(gymnasium.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_step += 1
        
        # Forzamos que no termine hasta llegar a max_steps
        if self.current_step < self.max_steps:
            terminated = False
            truncated = False
        else:
            truncated = True # Terminamos por límite de tiempo
            
        return obs, reward, terminated, truncated, info


#####################
# Inicio del Script Principal
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

# Creamos entorno
env = RlOnEnv()
obs, info = env.reset()

# Definimos Espacios para Gymnasium
# Acción: Elegir una ruta entre 0 y K-1
env.action_space = spaces.Discrete(K_PATHS)

# Observación: Vector
# Tamaño = (K_PATHS * TOTAL_SLOTS) + K_PATHS (utilización) + 1 (req slots)
input_size = (K_PATHS * TOTAL_SLOTS) + K_PATHS + 1
env.observation_space = spaces.Box(low=0, high=1, shape=(input_size,), dtype=np.float32)

# Asignamos recompensa y estado
env.setRewardFunc(get_reward)
env.setStateFunc(get_state)

# Cargamos topología de red
env.initEnviroment(fileDirectory + "/Biobio_Net.json", 
                   fileDirectory + "/Biobio_Net_routes.json", 
                   fileDirectory + "/Biobio_Net_bitrate.json")

# Configurar Simulador para entrenamiento
env.getSimulator().setGoalConnections(TOTAL_CONNECTIONS)
env.getSimulator().setLambda(160) 
env.getSimulator().setMu(1)
env.getSimulator().setAllocator(rl_agent_allocator) 
env.getSimulator().setConfidence(0.95)

# Fijamos semillas
env.getSimulator().setSeedArrive(42)
env.getSimulator().setSeedBitRate(43)
env.getSimulator().setSeedDeparture(44)
env.getSimulator().setSeedDst(45)
env.getSimulator().setSeedSrc(46)

# Inicializar Simulador
env.getSimulator().init()
env.start(verbose=False)

# Envolvemos el entorno
# Esto obliga a Gym a ignorar el 'done' prematuro del simulador y ke decimos que un episodio dura 1000 pasos 
STEPS_PER_EPISODE = 500
env = ForceLongEpisodeWrapper(env, max_steps=STEPS_PER_EPISODE)

print("INICIANDO ENTRENAMIENTO PPO")
print(f"Estado: Vector de tamaño {input_size}")
print(f"Acción: Seleccionar ruta 0 a {K_PATHS-1}")


# Arquitectura de la Red Neuronal (Policy Network)
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=[dict(pi=16*[64], vf=16*[64])] # capas y neuronas
)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.0003, 
    gamma=0.99, 
    gae_lambda=0.95, 
    clip_range=0.1,
    ent_coef=0.001, 
    batch_size=64, 
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_tensorboard/"
)

# Entrenar
print(f"Entrenando ...")
model.learn(total_timesteps=5000, progress_bar=True)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nTiempo de entrenamiento: {elapsed_time/60.} minutos")

# Guardar Modelo
model_name = "hermanoptico_v5"
model.save(model_name)
print(f"Modelo guardado como {model_name}.zip")