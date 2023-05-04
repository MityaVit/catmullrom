import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import interp1d

# Определение уравнения для графика гармонических колебаний

def f(x):
    return 2*np.sin(x) + 1.5*np.sin(2*x)

# Определение частоты и диапазона x

sr = 30
x = np.arange(0, 2*np.pi, 2*np.pi/(sr-1))

# Оценка графика гармонических колебаний

y = f(x)
fig, ax = plt.subplots(figsize=(8,4))

# Определение функции кривой Catmull-Rom

def catmull_rom_spline(points):

    # Расчёт сплайна Catmull-Rom

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    t = range(len(points))
    alpha = 0.5

    # Расчёт тангенсов каждой точки

    dt = np.gradient(t)
    dx = np.gradient(x)
    dy = np.gradient(y)
    m = np.sqrt(dx**2 + dy**2)
    tx = dx/m
    ty = dy/m

    # Расчёт вторых производных

    dtx = np.gradient(tx)/dt
    dty = np.gradient(ty)/dt

    # Расчёт контрольных точек

    px = x[1:-1] - alpha*tx[1:-1]*m[1:-1]/2 - alpha*tx[:-2]*m[:-2]/2
    py = y[1:-1] - alpha*ty[1:-1]*m[1:-1]/2 - alpha*ty[:-2]*m[:-2]/2
    qx = x[1:-1] + alpha*tx[1:-1]*m[1:-1]/2 + alpha*tx[:-2]*m[:-2]/2
    qy = y[1:-1] + alpha*ty[1:-1]*m[1:-1]/2 + alpha*ty[:-2]*m[:-2]/2

    # Интерполяция кривой с использованием контрольных точек

    curve_x = interp1d(t, x, kind='cubic')
    curve_y = interp1d(t, y, kind='cubic')
    curve_t = np.linspace(0, len(points)-1, 100)
    curve_points = np.column_stack((curve_x(curve_t), curve_y(curve_t)))

    return curve_points
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import interp1d

# Определение уравнения для графика гармонических колебаний

def f(x):
    return 2*np.sin(x) + 1.5*np.sin(2*x)

# Определение частоты и диапазона x

sr = 30
x = np.arange(0, 2*np.pi, 2*np.pi/(sr-1))

# Оценка графика гармонических колебаний

y = f(x)
fig, ax = plt.subplots(figsize=(8,4))

# Определение функции кривой Catmull-Rom

def catmull_rom_spline(points):

    # Расчёт сплайна Catmull-Rom

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    t = range(len(points))
    alpha = 0.5

    # Расчёт тангенсов каждой точки

    dt = np.gradient(t)
    dx = np.gradient(x)
    dy = np.gradient(y)
    m = np.sqrt(dx**2 + dy**2)
    tx = dx/m
    ty = dy/m

    # Расчёт вторых производных

    dtx = np.gradient(tx)/dt
    dty = np.gradient(ty)/dt

    # Расчёт контрольных точек

    px = x[1:-1] - alpha*tx[1:-1]*m[1:-1]/2 - alpha*tx[:-2]*m[:-2]/2
    py = y[1:-1] - alpha*ty[1:-1]*m[1:-1]/2 - alpha*ty[:-2]*m[:-2]/2
    qx = x[1:-1] + alpha*tx[1:-1]*m[1:-1]/2 + alpha*tx[:-2]*m[:-2]/2
    qy = y[1:-1] + alpha*ty[1:-1]*m[1:-1]/2 + alpha*ty[:-2]*m[:-2]/2

    # Интерполяция кривой с использованием контрольных точек

    curve_x = interp1d(t, x, kind='cubic')
    curve_y = interp1d(t, y, kind='cubic')
    curve_t = np.linspace(0, len(points)-1, 100)
    curve_points = np.column_stack((curve_x(curve_t), curve_y(curve_t)))

    return curve_points
p = np.polyfit(ref_points_x, ref_points_y, order)

    # Оценка полинома по всему диапазону x

    curve_points = np.column_stack((x, np.polyval(p, x)))
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'm--')

    # Расчёт ошибки восстановления

    interp_func = interp1d(curve_points[:, 0], curve_points[:, 1], kind='cubic')
    error = np.abs(interp_func(x) - y)
    max_error = np.max(error)

    # Вывод ошибки восстановления

    print(f"Максимальная ошибка восстановления: {max_error}")
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'm--', label=f'Кривая на основе полинома порядка {order}')

    ax.legend()
    fig.canvas.draw_idle()

    # Определение функции создания кнопки (просто для очистки графа)

def on_button3_click(event):
    ax.clear()
    fig.canvas.draw_idle()

    # Создание кнопок

ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'График')
button.on_clicked(on_button_click)

ax_button1 = plt.axes([0.6, 0.025, 0.1, 0.04])
button1 = Button(ax_button1, 'Catmull-Rom')
button1.on_clicked(on_button1_click)

ax_button2 = plt.axes([0.4, 0.025, 0.15, 0.04])
button2 = Button(ax_button2, 'Полином')
button2.on_clicked(on_button2_click)

ax_button3 = plt.axes([0.2, 0.025, 0.15, 0.04])
button3 = Button(ax_button3, 'Очистить')
button3.on_clicked(on_button3_click)

plt.subplots_adjust(bottom=0.2)
plt.show()
