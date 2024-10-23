import numpy as np
import matplotlib.pyplot as plt

# Definir el sistema de ecuaciones diferenciales
def f(t, y1, y2):
    y1_prime = y2
    y2_prime = 2 * y1 - 2 * y2 + np.exp(2 * t) * np.sin(t)
    return y1_prime, y2_prime

# Método de Euler para el sistema de ecuaciones
def euler_system(t0, tf, h, y1_0, y2_0):
    t_values = np.arange(t0, tf + h, h)
    y1_values = [y1_0]
    y2_values = [y2_0]
    
    for t in t_values[:-1]:
        y1 = y1_values[-1]
        y2 = y2_values[-1]
        
        y1_prime, y2_prime = f(t, y1, y2)
        
        y1_next = y1 + h * y1_prime
        y2_next = y2 + h * y2_prime
        
        y1_values.append(y1_next)
        y2_values.append(y2_next)
    
    return t_values, y1_values, y2_values

# Parámetros iniciales
t0 = 0
tf = 0.3
h = 0.01
y1_0 = -0.4  # y(0)
y2_0 = -0.6  # y'(0)

# Aplicar el método de Euler
t_values, y1_values, y2_values = euler_system(t0, tf, h, y1_0, y2_0)

# Mostrar los resultados
print("Resultados del método de Euler:")
for t, y1, y2 in zip(t_values, y1_values, y2_values):
    print(f"t = {t:.1f}, y ≈ {y1:.5f}, y' ≈ {y2:.5f}")

# Graficar la solución aproximada
plt.style.use('seaborn-darkgrid')  # Estilo mejorado
plt.figure(figsize=(10, 6))  # Tamaño de la figura

plt.plot(t_values, y1_values, label="y(t) (Euler)", color='blue', marker='o')
plt.plot(t_values, y2_values, label="y'(t) (Euler)", linestyle='--', color='orange', marker='x')

# Etiquetas y título
plt.xlabel('Tiempo (t)', fontsize=14)
plt.ylabel('Valores de y', fontsize=14)
plt.title('Solución usando el método de Euler', fontsize=16)
plt.grid(True)

# Leyenda
plt.legend(fontsize=12)

# Mostrar la gráfica
plt.show()
