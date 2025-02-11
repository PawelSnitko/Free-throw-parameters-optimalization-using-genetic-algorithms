import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# --- Stałe fizyczne ---
GRAVITY = 9.81  # m/s²
BASKET_HEIGHT = 3.05  # Wysokość kosza w metrach
BASKET_DISTANCE = 4.57  # Odległość od kosza w metrach
BALL_RADIUS = 0.12  # Promień piłki w metrach
BASKET_RADIUS = 0.225  # Promień obręczy kosza w metrach
PLAYER_HEIGHT = 1.85  # Wysokość rąk gracza w metrach
BACKBOARD_WIDTH = 1.8  # Szerokość tablicy kosza w metrach
BACKBOARD_HEIGHT = 1.05  # Wysokość tablicy kosza w metrach
BACKBOARD_X = BASKET_DISTANCE + 0.15  # Współrzędna X tablicy kosza

# --- Typy obręczy ---
rim_types = {
    "Hard Rim": 0.95,  # Współczynnik odbicia 90%
    "Medium Rim": 0.5,  # Współczynnik odbicia 70%
    "Soft Rim": 0.3,  # Współczynnik odbicia 50%
}

# --- Funkcja symulująca rzut piłki ---
def simulate_shot(angle, speed, rotation, rim_type):
    angle_rad = np.radians(angle)
    time_of_flight = (2 * speed * np.sin(angle_rad)) / GRAVITY
    t = np.linspace(0, time_of_flight, num=1000)

    x = speed * np.cos(angle_rad) * t
    y = PLAYER_HEIGHT + speed * np.sin(angle_rad) * t - 0.5 * GRAVITY * t**2

    for i, (xi, yi) in enumerate(zip(x, y)):
        # Sprawdzenie kontaktu z obręczą
        if (xi - BASKET_DISTANCE)**2 + (yi - BASKET_HEIGHT)**2 <= (BASKET_RADIUS + BALL_RADIUS)**2:
            # Zmiana prędkości i kierunku odbicia
            speed *= rim_type
            rebound_angle = np.radians(random.uniform(-30, 30))  # Losowy kąt odbicia
            new_angle = angle_rad + rebound_angle

            # Oblicz nową trajektorię
            t_rebound = np.linspace(0, (2 * speed * np.sin(new_angle)) / GRAVITY, num=500)
            x_rebound = xi + speed * np.cos(new_angle) * t_rebound
            y_rebound = yi + speed * np.sin(new_angle) * t_rebound - 0.5 * GRAVITY * t_rebound**2

            # Czy piłka wpada do kosza po odbiciu?
            for x_new, y_new in zip(x_rebound, y_rebound):
                if (x_new - BASKET_DISTANCE)**2 + (y_new - BASKET_HEIGHT)**2 <= BASKET_RADIUS**2 and y_new < BASKET_HEIGHT:
                    return x, y  # Piłka wpada po odbiciu
            return x, y  # Piłka wypadła z kosza
    return x, y  # Trajektoria bez trafienia

# --- Zmodyfikowana funkcja celu ---
def objective_function(individual, rim_type):
    angle, speed, rotation = individual
    x, y = simulate_shot(angle, speed, rotation, rim_type)

    for i, (xi, yi) in enumerate(zip(x, y)):
        if (xi - BASKET_DISTANCE)**2 + (yi - BASKET_HEIGHT)**2 <= (BASKET_RADIUS + BALL_RADIUS)**2:
            if yi < BASKET_HEIGHT:
                return (1.0,)  # Trafienie
    return (-abs(x[-1] - BASKET_DISTANCE),)  # Kara za nietrafienie (odległość od kosza)

# --- Konfiguracja algorytmu genetycznego ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_angle", random.uniform, 30, 60)  # Zakres kątów rzutu
toolbox.register("attr_speed", random.uniform, 5, 8)   # Zakres prędkości początkowej
toolbox.register("attr_rotation", random.uniform, 0, 5)  # Zakres rotacji piłki
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_angle, toolbox.attr_speed, toolbox.attr_rotation), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def run_simulation():
    colors = {"Hard Rim": "red", "Medium Rim": "green", "Soft Rim": "blue"}
    plt.figure(figsize=(10, 6))

    for rim_name, rim_coefficient in rim_types.items():
        toolbox.register("evaluate", objective_function, rim_type=rim_coefficient)
        toolbox.register("mate", tools.cxBlend, alpha=0.3)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=100)
        NGEN = 50

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        logbook = tools.Logbook()

        population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=NGEN, stats=stats, verbose=False)

        best_ind = tools.selBest(population, k=1)[0]
        print(f"[{rim_name}] Najlepsze parametry rzutu: Kąt: {best_ind[0]:.2f}°, Prędkość: {best_ind[1]:.2f} m/s, Rotacja: {best_ind[2]:.2f} obr/s")

        x, y = simulate_shot(best_ind[0], best_ind[1], best_ind[2], rim_coefficient)
        plt.plot(x, y, label=f"{rim_name}", color=colors[rim_name])

    # Rysowanie obręczy kosza
    obr_x_start = BASKET_DISTANCE - BASKET_RADIUS
    obr_x_end = BASKET_DISTANCE + BASKET_RADIUS
    plt.plot([obr_x_start, obr_x_end], [BASKET_HEIGHT, BASKET_HEIGHT], color='orange', linewidth=3, label="Obręcz kosza")

    # Rysowanie tablicy kosza
    backboard_x = [BACKBOARD_X, BACKBOARD_X]
    backboard_y = [BASKET_HEIGHT - BACKBOARD_HEIGHT / 2, BASKET_HEIGHT + BACKBOARD_HEIGHT / 2]
   

    plt.xlabel('Odległość (m)')
    plt.ylabel('Wysokość (m)')
    plt.legend()
    plt.title("Trajektorie rzutów z uwzględnieniem tablicy kosza")
    plt.grid()
    plt.show()

# --- Uruchomienie symulacji ---
run_simulation()
