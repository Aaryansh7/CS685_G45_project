# CS685 project IITK

##### This repository contains programs for making detailed catalogues, finding interesting patterns and detecting maneuvers among space objects, from the raw TLE data available online. The repository has been structured into different modules depending on the type of tasks they perform. Following is the strucutre :-
1) Error analysis(sgp4) :-  Contains programs that use sgp4 python library to predict positions, velocites and analyse error propagation from TLE data. 
2) Catalogue_from_tles - Contains programs that are able to make detailed catalgues from online TLE data.
3) Clustering - This folder conatins programs that perform a variety of functions. Some of them are like maneuver detections, Clustering in similar groups etc.
4) Debris predictions - Classifier for predicting an celestial object as Space Debris or satellite.
5) Maneuver Detections - Algorithms to detect possible maneuvers done for an active satellite.


### Requirements:-
1)sgp4
2)skyfield
3)tqdm
4)numpy
5)matplotlib
6)csv
7)xlsx
8)pandas

More description can be found in the documenation files of these respective folders. 





