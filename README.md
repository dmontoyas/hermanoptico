# Hermanóptico: Agente de Aprendizaje por Refuerzo para Asignación Dinámica de Rutas en Redes Ópticas Elásticas

Requerimientos:
- Python = 3.10.6
- Stable-Baselines3 = 2.7.0
- Tensorflow = 2.15.0
- Protobuf = 4.25.8
- Gymnasium = 0.28.1
- mpi4py = 4.1.0
- [DREAM-ON GYM](https://gitlab.com/IRO-Team/dream-on-gym-v2.0) = 0.0.9

Información:
- sp-ff.py : Implementa heurísticas Shortest Path y First Fit en red óptica simulada Biobio_Net
- entrenar_agente.py : Script que entrena agente sobre red óptica simulada
- entorno_pruebas.py : Script para calcular probabilidad de bloqueo sobre distintos escenarios de tráfico de red para agente ya entrenado.
- Biobio_Net.json Biobio_Net_bitrate.json Biobio_Net_routes.json : Archivos que contienen información sobre la topología de red necesarios para implementar la simulación.
