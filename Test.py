import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Teil 1: Generiere 100 zufällige 2-Tupel (x, y), wobei x und y ∈ [1, 10]
data = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(100)]

# Teil 2: Bestimme Klassen basierend auf der Attributsumme
labels = []
sums = []
for x, y in data:
    s = x + y
    sums.append(s)
    if 9 <= s <= 13:
        labels.append("A")
    else:
        labels.append("B")

# a) Histogramm der Attributssummen
plt.figure(figsize=(8, 5))
plt.hist(sums, bins=range(2, 22), edgecolor='black')
plt.title("Histogramm der Attributssummen")
plt.xlabel("Summe von Attributen")
plt.ylabel("Häufigkeit")
plt.grid(True)
plt.show()

# b) Visualisiere die Daten farblich nach Klassen
colors = ['blue' if label == 'A' else 'red' for label in labels]
x_vals, y_vals = zip(*data)

plt.figure(figsize=(8, 5))
plt.scatter(x_vals, y_vals, c=colors, label='Datenpunkte')
plt.title("Visualisierung der Tupel nach Klassen")
plt.xlabel("Attribut 1")
plt.ylabel("Attribut 2")
plt.grid(True)
plt.show()

# c) Aufteilen in Trainings- und Testmenge
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Visualisiere Trainings- und Testdaten
def visualize_split(X, y, title):
    x_vals, y_vals = zip(*X)
    colors = ['blue' if label == 'A' else 'red' for label in y]
    plt.scatter(x_vals, y_vals, c=colors)
    plt.title(title)
    plt.xlabel("Attribut 1")
    plt.ylabel("Attribut 2")
    plt.grid(True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
visualize_split(X_train, y_train, "Trainingsdaten")

plt.subplot(1, 2, 2)
visualize_split(X_test, y_test, "Testdaten")

plt.tight_layout()
plt.show()