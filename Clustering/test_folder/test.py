from skyfield.api import load
from skyfield.elementslib import osculating_elements_of

ts = load.timescale()
t = ts.utc(2018, 4, 22, 0)

planets = load('https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/spk/de421.bsp')
earth = planets['earth']
moon = planets['moon']

position =  earth.at(t)
print(moon.at(t).position.km)
elements = osculating_elements_of(position)