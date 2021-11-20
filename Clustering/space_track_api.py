from datetime import datetime
import configparser
import json
import pandas as pd
import requests
import time
import xlsxwriter
import urllib
#from urlparse import urljoin 
class MyError(Exception):
    def __init___(self,args):
        Exception.__init__(self,"my exception was raised with arguments {0}".format(args))
        self.args = args


uriBase                = "https://www.space-track.org"
requestLogin           = "/ajaxauth/login"
requestCmdAction       = "/basicspacedata/query"

# Use configparser package to pull in the ini file (pip install configparser)
config = configparser.ConfigParser()
config.read("../data_files/SLTrack.ini")
configUsr = config.get("configuration","username")
configPwd = config.get("configuration","password")
siteCred = {'identity': configUsr, 'password': configPwd}

def getTLESatellites(Class, Orderby, Sort, Limit, Format,
                     Predicate1, Operator1, Value1, Predicate2, Operator2, Value2):

    with requests.Session() as session:
        # run the session in a with block to force session to close if we exit
        # need to log in first. note that we get a 200 to say the web site got the data, not that we are logged in
        resp = session.post(uriBase + requestLogin, data = siteCred)
        if resp.status_code != 200:
            raise MyError(resp, "POST fail on login")

        # this query picks up TLEs for all satellites from the catalog. Note - a 401 failure shows you have bad credentials
        # Enter the fields for which query must be built

        query_link = ("/class/" + Class + "/" + Predicate1 + "/" + Value1
                        + "/" + Predicate2 + "/" + Operator2 + Value2 + "/orderby/" + Orderby + " " + Sort + "/" + 
                        "limit/" + Limit + "/" + "format/" + Format + "/emptyresult/show")

        #query_link = urllib.parse.quote(query_link, safe='')
        #print(query_link)
        #URL=uriBase + requestCmdAction + query_link
        URL= urllib.parse.urljoin(uriBase, requestCmdAction)
        URL= urllib.parse.urljoin(URL, query_link)
        #print (URL)
        URLencode= urllib.parse.quote(URL)
        #print(URLencode)

        resp = session.get(uriBase + requestCmdAction + query_link)
        if resp.status_code != 200:
            print(resp)
            print(query_link)
            print(resp.text)

            raise MyError(resp, "GET fail on request for satellites")

        # resp.text contains the required Tles
        data = resp.text
        #file_object = open('../data_files/temp.tle', 'w')
        #file_object.write(data)
        #file_object.close()
        session.close()
    return data

'''
if __name__ == "__main__":
    Class = 'tle_latest'
    Orderby = 'EPOCH'
    Sort = 'desc'
    Limit = '5'
    Format = '3le'
    Predicate1 = 'NORAD_CAT_ID'
    Operator1 = ''
    Value1 = '46674'
    Predicate2 = 'EPOCH'
    Operator2 = '<'
    Operator2 = urllib.parse.quote(Operator2, safe='')
    print(Operator2)
    Value2 = '2021-05-27 23::00:00'
    Value2 = urllib.parse.quote(Value2, safe='')
    print(Value2)
    data = getTLESatellites(Class, Orderby, Sort, Limit, Format,
                     Predicate1, Operator1, Value1, Predicate2, Operator2, Value2)
    print(data)

'''
