import pandas as pd
import json
'''
1.       Bench, sFoLqgI5afw6UcfrmxZX
2.       Barbell Curl, bkziJY2HV6G0NF0zUUqU
3.       Back Squat, lCcTdGNrsSUCUcWFpdHh
4.       Overhead Press, 9PJxOlCdxcbFBvZw7gk9
5.       Deadlift, fZXPgJEMjxIMjz5ZxXqZ

'''


with open("data.json") as f:

    data = json.load(f)

    # Overhead Press, 9PJxOlCdxcbFBvZw7gk9
    overhead_xacc = list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['xacc'])
    overhead_yacc = list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['yacc'])
    overhead_zacc = list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['zacc'])
    overhead_time = list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_time']['gu5M31wNhf8D8Z85y7j9']['time'])

    # Barbell Curl, bkziJY2HV6G0NF0zUUqU
    curl_xacc = list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['xacc'])
    curl_yacc = list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['yacc'])
    curl_zacc = list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['zacc'])
    curl_time = list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_time']['lmVhWYwzYvQiVD1zGUMv']['time'])
    

    # Deadlift, fZXPgJEMjxIMjz5ZxXqZ
    deadlift_xacc = list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['xacc'])
    deadlift_yacc = list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['yacc'])
    deadlift_zacc = list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['zacc'])
    deadlift_time = list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_time']['aHWqtDacLbu2GfV68fYI']['time'])
    
    
    # Back Squat, lCcTdGNrsSUCUcWFpdHh
    squat_xacc = list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['xacc'])
    squat_yacc = list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['yacc'])
    squat_zacc = list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['zacc'])
    squat_time = list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_time']['aFWme3iCwy1vX7K4cTot']['time'])
   

    # Bench, sFoLqgI5afw6UcfrmxZX
    bench_xacc = list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['xacc'])
    bench_yacc = list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['yacc'])
    bench_zacc = list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['zacc'])
    bench_time = list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_time']['aFWme3iCwy1vX7K4cTot']['time'])
print("done reading...")