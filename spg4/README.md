# SGP4

##### This folder contains programs that use sgp4 python library to predict positions, velocites and analyse error propagation from TLE data. There are programs that show graphical varaiations in positons and velocitiees across tles as calculated by the spg4 library.Following is the structure:


1. sgp4_err.py  - This program shows the analysis of errors in position and velocities across tles of the same object. For running the program within the same directory - 
```
python3 sgp4_err.py 
```
2. tle_errprop.py - This program analyses the error in a single or specific TLE propagation over time. For running the program within the same directory - 
```
python3 tle_errprop.py 
```
3. ric_err.py  - Sgp4 mthod by default computes the position and velocities in TEME coordinate frame. However analysis in sifferent coordinate frames such as ECI, RIC can provide you interesting observations in diiferent axises. This program solves the same purpose. For running the program within the same directory - 
```
python3 ric_err.py 
```
4. tle_avgerr.py - This program analyses how, and at what level of magnitude, the errors are propagting. For running the program within the same directory - 
```
python3 tle_avgerr.py 
```
