import numpy as np

ticks_per_meter_measurements = (178 / 0.09, 100 / 0.05, 140 / 0.07)
TICKS_PER_METER = np.average(ticks_per_meter_measurements)

# CONSTANTS
CART_MASS = 217.88e-3  # 4 magnets.
MASS_ERROR = 1e-5  # 0.01g

length = 9.8 / 100
length_error = 0.1 / 100
g = 9.81
g_error = 0.1
k1_mass = 42.42e-3
k1 = (k1_mass * g) / length  # Red spring at left side of the  cart.
k1_error = np.sqrt(
    ((k1_mass / length) * g_error) ** 2
    + ((k1_mass * g) * length_error * (1 / length**2)) ** 2
    + ((g / length) * MASS_ERROR) ** 2
)
k2_mass = 43e-3
k2 = (k2_mass * g) / length
k2_error = np.sqrt(
    ((k2_mass / length) * g_error) ** 2
    + ((k2_mass * g) * length_error * (1 / length**2)) ** 2
    + ((g / length) * MASS_ERROR) ** 2
)
SPRING_CONSTANT = k1 + k2
SPRING_CONSTANT_ERROR = np.hypot(k1_error, k2_error)
NATURAL_FREQUENCY = np.sqrt(SPRING_CONSTANT / CART_MASS)
NATURAL_FREQUENCY_ERROR = np.hypot(
    (1 / CART_MASS) * SPRING_CONSTANT_ERROR,
    SPRING_CONSTANT * MASS_ERROR * (1 / CART_MASS**2),
)
