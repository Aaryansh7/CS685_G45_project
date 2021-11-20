# Clustering

This folder conatins programs that perform a variety of functions. Some of them are like maneuver detections, Clustering in similar groups etc.
The purpose of the programs can be broadly categorised into Three parts - 
1. Orbit Similarities
2. Maneuver detections
3. Arrangements in Orbits

### 1. Orbit Similarities
Satellites sharing the same or similar orbit(s) are most likely to be related either in ownership, intent or function. Finding these satellites, their relationships and distribution is key to obtaining a first guess of their intent and function. At times, satellites with clear similarity in orbits may be launched from different locations, with very different (official) names, by different space actors; but be performing activities on concert with each other. The objective here, is to detect these relationships purely based on publicly available tracking data to build the initial hypothesis, and pass on data in structured form for further analysis.

With the knowledge of semi-major axis (SMA), we partition the near-Earth RSOs into multiple shells:
Tight altitude clusters containing objects within 2km of a notional shell SMA - outliers from which are passed to the next step
Loose altitude clusters containing objects within 10km of a notional SMA - outliers from which are passed to the next step
Broad altitude groups containing objects within 25km of a notional SMA
Outliers from the above, last step are marked as such to denote no relationships with other objects

For each altitude shell defined by a centroid SMA and accompanying set of RSOs, we can extract unique clusters of objects based on the planes they inhabit. Since the orbital plane is defined by the 2 angular parameters of inclination and Right Ascension of Ascending Node, clusters of objects with similar [i, RAAN] pairs can be considered to be in the same plane. As an extension, planes sharing the same value of inclination, but with RAANs displaying a pattern of {RAANj = RAANi + 360/n}, for every RAANj > RAANi, and where n = number of planes sharing the same inclination, within the same altitude shell.

There are multiple programs that link together to perform the above tasks. We have described in the sequence that they follow.
1. filter.py - This program simply removes objects from recent elset data that are of no concern, such as Debrisis, or satellites that are non-functional etc.
2. kmeans+dbscan_catalogue.py - This program clusters the filtered objects based on their sma values in order to get the 3 types of altitude based clustering as mentioned in the documentation above. Kmeans clustering is performed for grouping based on sma values.
3. add_norad_id_kmeans_data.py - In order to reduce computation, only some of the fields from the detailed catlogue were used in the above programs. Therefore, this program simply adds the rest of the required fields to the clustered data.
4. cluster_raan_inclination.py - This program finds clusters based on inc, raan values for objects that within sma labels, i-e objects within same altitude groups. The Algorithm used is DBSCAN for optimal results.

Run the program within same directory as follows:- 
```
python3 filter.py
```
```
python3 kmeans+dbscan_catalogue.py
```
```
python3 add_norad_id_kmeans_data.py
```
```
python3 cluster_raan_inclination.py
```
### 2. Maneuver detections
This comprises of SMA based maneuver detections and inclination based maneuver detections. Following are the programs:-
1. sma_maneuver_detect.py - This program detects or predicts the maneuver epochs based on the TLE data of the object of concern. We have used a stastical based method to predict sma_based maneuvers. We first filter the data such tha we could have a smooth graph of this data. This would help in removing out possible unwanted or noisy elemnts in TLE samples.
In case of actual maneuvers, we had observed that sma values generally increase much more steeply than in case of naturall changes. Therefore, we have used certain thresholds of slopes to predict such type of maneuvers.
Follow the command to run the program within same directory - 
```
python3 sma_maneuver_detect.py
```

2. inc_maneuver_detect.py - This program detects inclination based maneuvers. The methodology used is very much similar to that of sma maneuver detections. However detailed information about this can be found at last of this documentation.
Follow the command to run the program within same directory - 
```
python3 sma_maneuver_detect.py
```

3. Arrangements in orbits
RSOs, working together, when sharing the same orbit (SMA, i, RAAN) may or may not be actively controlling their relative positions. Based on this all patterns observed will fall under one of the following types:
- No control at all (free dispersion, evolution)
- Passive management
  - One-time: Controlled/timed deployment from launcher upper stage
  - Non-propulsive passive management at individual RSO level
- Active propulsive management at individual RSO level

For both the passively & actively managed intra-plane intra-altitude RSO groups, an analyst needs to look for all types of persistent patterns, including (but limited) to the following:
- Equal angular spacing, covering the full orbit
- Equal spacing, without covering the full orbit
- Equal spacing, with 1 or more RSOs either misplaces or missing from their designated slots
- Tight formations: pairs, trios, quads
A pattern can be termed 'persistent' if tracking data shows consistent effort being expended to maintain a particular pattern for the same set of RSOs. This aspect can be clearly deduced through a short time-history for the same group, thereby eliminating instances where freely evolving geometries show patterns at singular points in time.

The program ``` arrangements_orbits.py ``` solves the same purpose. It identifies the above mentioned patterns or groupings.


### Note-
All the data generated files can be found in the data_files folder.


## Documentation for Maneuver detections
### Inclination based maneuver detections:(specifically for those type of maneuvers in which inclination is increased): 
The purpose of this exercise was to find maneuver dates from TLE data of space objects.

We have used a statistical based method to detect these types of maneuvers. The input is the inclination values of the object of concern and the corresponding epochs. Following is our methodology used:
We analyzed the data in the form of an “Inclination-vs-Epoch” graph. We observed that the maneuver dates (as per the ground truth data) were occurring near to significant peaks in the graph. However there were many peaks that had small abrupt changes to which were not the actual maneuver dates. Therefore, the data needed to be refined in order to remove such noises. We used Savitzky–Golay filter  to smoothen out the graph, thus removing noisy elements. The parameters are 101 for window size and 13 for polynomial order. These parameters have been decided based on optimal response obtained by tuning.

The general trend is to do an inclination based maneuver whenever the inclination has to be increased. This increase can be discovered when the average of inclination values of a set threshold number of next consecutive data points from the current data pointer is greater than the current data pointer’s inclination value. The magnitude of threshold and the averages depend on how the data has been refined. Different results would be obtained depending upon the type of refinement.  For example if one uses Savitzky-Golay filter or even if no filter has been used, different thresholds would have to be set for each type respectively.  Along with this condition we have added that, inclination values of say 4-5 next consecutive data pointers are in ascending order.
By doing this, one would obtain many consecutive dates around the actual maneuver dates. To find out unique dates out of these, we took the median of these consecutive dates as the predicted maneuver date.

To test our performance, we compared the ground truth data with the TLE data of Envisat and other few satellites with ground truth available. Labels were created for those dates in TLE data that were nearest to the predicted maneuver dates( a threshold can be defined for this). The performance score was then calculated by a confusion matrix.
The performance score(F1 score) was 0.88 for Envisat.

We would look further in improving this performance.







