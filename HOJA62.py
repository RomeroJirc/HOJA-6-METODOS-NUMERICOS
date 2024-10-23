import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
a, b = 0, 1  # Intervalo [a, b] actualizado hasta 1
h = 0.1  # Tamaño del paso
t_values = np.arange(a, b + h, h)  # Valores de t

# Condiciones iniciales
y1_0 = 0  # y(0) = 0
y2_0 = 0  # y'(0) = 0

# Valores exactos proporcionados
exact_values = {
    0.0: 0.0,
    0.1: 8.94e-6,
    0.2: 1.535e-4,
    0.3: 1.535e-4,
    0.4: 2.831e-3,
    0.5: 7.431e-3,
    0.6: 0.01656,
    0.7: 0.032998,
    0.8: 0.0605661,
    0.9: 0.104405,
    1.0: 0.1713287
}

# Función para obtener el valor exacto de y(t)
def get_exact_value(t):
    if t in exact_values:
        return exact_values[t]
    else:
        # Si el valor exacto no está en la lista, se usa interpolación lineal
        ts = sorted(exact_values.keys())
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i + 1]:
                t1, y1 = ts[i], exact_values[ts[i]]
                t2, y2 = ts[i + 1], exact_values[ts[i + 1]]
                return y1 + (y2 - y1) * (t - t1) / (t2 - t1)
        return None  # Por si no se encuentra

# Inicialización de listas para almacenar resultados
y1_values = [y1_0]
y2_values = [y2_0]

# Método de Euler
print("Paso | t    | Aproximación (y1) | Valor Exacto | Error")
print("-" * 50)
for i in range(1, len(t_values)):
    t = t_values[i - 1]
    y1 = y1_values[-1]
    y2 = y2_values[-1]

    # Derivadas
    y1_prime = y2
    y2_prime = 2*y2 - y1 + t*np.exp(t) - t

    # Actualización de valores usando el método de Euler
    y1_next = y1 + h * y1_prime
    y2_next = y2 + h * y2_prime

    # Almacenar los resultados
    y1_values.append(y1_next)
    y2_values.append(y2_next)

    # Valor exacto y cálculo del error
    exact = get_exact_value(t + h)
    error = abs(y1_next - exact)

    # Imprimir resultados
    print(f"{i:3}  | {t + h:.2f} | {y1_next:.6f}          | {exact:.6f}    | {error:.6f}")

# Cálculo del error usando los valores exactos proporcionados
errors = [abs(y1_values[i] - get_exact_value(t_values[i])) for i in range(len(t_values))]

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(t_values, y1_values, label='Aproximación (Euler)', marker='o')
plt.scatter(t_values, [get_exact_value(t) for t in t_values], color='red', label='Valores Exactos', zorder=5)
plt.title('Aproximación de la solución usando el Método de Euler')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el error
plt.figure(figsize=(12, 6))
plt.plot(t_values, errors, label='Error de aproximación', marker='x', color='red')
plt.title('Error de Aproximación en cada paso')
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()