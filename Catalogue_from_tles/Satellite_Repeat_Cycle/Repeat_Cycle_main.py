import ephem
import math
from datetime import date, datetime, timedelta
from sgp4.api import jday
from sgp4.api import Satrec
import numpy as np 

Radius_earth = 6373.0

print("entered")
# Get Ephemeris Data
def getEphem(satellite, date):
  satellite.compute(date)
  #print(satellite)
  latitude  = satellite.sublat / ephem.degree
  longitude = satellite.sublong / ephem.degree
  hight     = satellite.elevation
  mmotion   = satellite._n
  obperiod  = 1 / mmotion
  return (latitude, longitude, hight, mmotion, obperiod)

def jday2str(jday):
    (year, month, day, hour, minute, second) = ephem.Date(jday).tuple()
    second = int(second)
    dt = datetime(year, month, day, hour, minute, second)
    return dt.isoformat().replace('T', ' ')

def distance_earth_points(lat0, long0, lat1, long1):
  lat0 = math.radians(lat0)
  lon0 = math.radians(long0)
  lat1 = math.radians(lat1)
  lon1 = math.radians(long1)

  dlon = lon1 - lon0
  dlat = lat1 - lat0

  a = math.sin(dlat / 2)**2 + math.cos(lat0) * math.cos(lat1) * math.sin(dlon / 2)**2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

  distance = Radius_earth * c

  return distance

def lat0date(satellite, date, lat, eps):
  #print("Function lat0date ....")
  d1 = date
  #print("d1 : " +str(d1))
  satellite.compute(d1)
  l1  = satellite.sublat / ephem.degree
  #print("l1 : " + str(l1))
 
  d2 = d1 + eps
  #print("d2 : " + str(d2))
  satellite.compute(d2)
  l2  =satellite.sublat / ephem.degree
  #print("l2 : " + str(l2))

  a = (l2 - l1) / eps
  if a == 0:
    a+=0.001
  b = l1 - a * d1

  dt   = (lat - b) / a
  
  return dt

def norm(x1, x2, x3):
  norm = np.sqrt(x1**2 + x2**2 + x3**2)
  return norm

# TLE: TLE of satellite
# datestr: Initial date (string)
# maxlat: max error of latitude[deg]
# maxlong: max error of longitude[deg]
# maxdays: max days of search

def searchRepeatCycle(TLE, datestr, maxlat, maxlong, maxdays):
    #print("Function searchRepeatCycle started...")

    min_distance = float("inf")
    Initial_Latitude_ = None
    Initial_Longitude_ = None
    Latitude_ = None
    Longitude_ = None
    Diff_Latitude = None
    Diff_Longitude = None
    Days_ = None
    Min_Distance = None


    # Initial Date
    dt0 = dt = ephem.Date(datestr)

    # Calculate Initial Ephemeris
    #(line1, line2, line3) = TLE.split("\n")
    (line1, line2, line3) = TLE
    satellite = ephem.readtle(line1, line2, line3)
    satellite_ = Satrec.twoline2rv(line2, line3)
    (latitude0, longitude0, hight0, mmotion0, obperiod0) = getEphem(satellite, dt)
    Initial_Latitude_ = latitude0
    Initial_Longitude_ = longitude0

    # Search Repeat Cycle
    eps      = obperiod0 / 360 / 10
    latlen   = 40009 # Circumference - meridional [Km]
    longlen  = 40075 # Circumference - quatorial  [Km]

    dt = dt0
    #print("                                         Lat(+N) diff[deg]   diff[Km] |   Long(+E)  diff[deg]   diff[Km] | Distance(Km)")
    for d in range(int(1000)):
      (latitude, longitude, _, _, _) = getEphem(satellite, dt)
      difflat   = latitude  - latitude0
      difflong  = longitude - longitude0

      dtstr =jday2str(dt)
      dt_ = datetime.strptime(dtstr, '%Y-%m-%d %H:%M:%S')
      jd_, fr_ = jday(dt_.year, dt_.month, dt_.day, dt_.hour, dt_.minute, dt_.second) #(year, month, date, hour, min, sec)
      e, r, v = satellite_.sgp4(jd_, fr_)
      position = norm(r[0], r[1], r[2])
      if position <= Radius_earth:
        Days_ = None
        break

      distance = distance_earth_points(latitude0,longitude0,latitude,longitude)
      if distance < min_distance and d>0:
        min_distance = distance
        #print(min_distance)
        Latitude_ = latitude
        Longitude_ = longitude
        Diff_Latitude = difflat
        Diff_Longitude = difflong
        Days_  = dt - dt0
        Min_Distance = min_distance

      if abs(difflat) < maxlat and abs(difflong) <maxlong:
        dtstr =jday2str(dt)
        days  = dt - dt0
        difflatlen  = difflat  / 360 * latlen
        difflonglen = difflong / 360 * longlen * math.cos(math.radians(latitude))
        #print("[%s = %6.2f(days)] %10.4f %+10.4f %+10.4f | %10.4f %+10.4f %+10.4f | %10.4f" % (dtstr, days, latitude, difflat, difflatlen, longitude, difflong, difflonglen, distance))

      dt = lat0date(satellite, dt + obperiod0, latitude0, eps)

    #print(min_distance)
    return Initial_Latitude_, Initial_Longitude_, Latitude_, Longitude_, Diff_Latitude, Diff_Longitude, Days_, Min_Distance

# Search Repeat Cycle for each TLE and datestr
def searchRepeatCycles(TLE, datestrs, maxlat, maxlong, maxdays):
    init_lat = None
    init_long = None
    lat_ = None
    long_ = None
    diff_lat = None
    diff_long = None
    days =  None
    min_distance = None

    init_lat, init_long, lat_, long_, diff_lat, diff_long, days, min_distance = searchRepeatCycle(TLE, datestrs, maxlat, maxlong, maxdays)
    #print("init lat: " + str(init_lat))

    return init_lat, init_long, lat_, long_, diff_lat, diff_long, days, min_distance
  
