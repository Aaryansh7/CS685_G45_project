import ephem
import math
from datetime import date, datetime, timedelta

Radius_earth = 6373.0

# Get Ephemeris Data
def getEphem(satellite, date):
  satellite.compute(date)
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



# TLE: TLE of satellite
# datestr: Initial date (string)
# maxlat: max error of latitude[deg]
# maxlong: max error of longitude[deg]
# maxdays: max days of search

def searchRepeatCycle(TLE, datestr, maxlat, maxlong, maxdays):
    print("Function searchRepeatCycle started...")

    #min_distance = #inf
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
    print("Initial date dt0: " + str(dt0))
    print("Initial date dt: " + str(dt))

    # Calculate Initial Ephemeris
    #(line1, line2, line3) = TLE.split("\n")
    (line1, line2, line3) = TLE
    satellite = ephem.readtle(line1, line2, line3)
    (latitude0, longitude0, hight0, mmotion0, obperiod0) = getEphem(satellite, dt)
    print("latitude0 : " + str(latitude0))
    print("longitude0 : " + str(longitude0))
    print("height0 : " + str(hight0))
    print("mmotion0 : " + str(mmotion0))
    print("obperiod0 : " + str(obperiod0))
    Initial_Latitude_ = latitude0
    Initial_Longitude_ = longitude0

    # Search Repeat Cycle
    eps      = obperiod0 / 360 / 10
    latlen   = 40009 # Circumference - meridional [Km]
    longlen  = 40075 # Circumference - quatorial  [Km]

    dt = dt0
    print("                                         Lat(+N) diff[deg]   diff[Km] |   Long(+E)  diff[deg]   diff[Km] | Distance(Km)")
    for d in range(int(530000)):
      (latitude, longitude, _, _, _) = getEphem(satellite, dt)
      difflat   = latitude  - latitude0
      difflong  = longitude - longitude0
      distance = distance_earth_points(latitude0,longitude0,latitude,longitude)
      if distance < min_distance:
        min_distance = distance
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

      # update dt
      (year, month, day, hour, minute, second) = ephem.Date(dt).tuple()
      dt = datetime(year, month, day, hour, minute, int(second))
      dt = dt + timedelta(seconds=60)
      datestr = str(dt)
      dt = ephem.Date(datestr)

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

  init_lat, init_long, lat_, long_, diff_lat, diff_long, days = searchRepeatCycle(TLE, datestr, maxlat, maxlong, maxdays)

  return init_lat, init_long, lat_, long_, diff_lat, diff_long, days, min_distance
   
