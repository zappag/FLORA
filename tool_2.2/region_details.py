# region_details.py
def get_region_details(watershed):
    if watershed == "panaro":
        #tlon=10.99
        #tlat=44.65
        #tlabel='modena'
        tlon=11.045
        tlat=44.727
        tlabel='bomporto'   
        hydrobasin_level='07' # watershed specific
        box=(44, 10.25, 45.2, 11.75)  # panaro only, lev 7
    elif watershed=="reno":
        #tlon=11.3
        #lat=44.49
        #tlabel='bologna'
        tlon=11.282
        tlat=44.480
        tlabel='casalecchio'
        hydrobasin_level='07' # watershed specific
        box=(43.9, 10.75, 44.9, 12.3)  # reno full, lev 7
        #box=(43.9, 10.75, 44.7, 11.4)  # reno Bologna, lev 8
    elif watershed=="timis":
        hydrobasin_level='07'
        tlon=22.14 # caransebes
        tlat=45.21
        tlabel='caransebes'
        box=(44.75, 20, 46.25, 23)  
    elif watershed=="lagen":
        hydrobasin_level='06'
        tlon=10.47 # lillehammer
        tlat=61.11 # 
        box=(60, 7, 62.5, 12) 
        tlabel='lillehammer'
    elif watershed=="aragon":
        hydrobasin_level='07'
        tlon=-1.1
        tlat=42.6
        tlabel='randompoint'
        box=(42, -2.5, 43.2, -0.25)
    else:
        print("Unknown watershed")
        exit(1) 


    return {
        'shape_file_level': hydrobasin_level,
        'tlat': tlat,
        'tlon': tlon,
        'tlabel': tlabel,
        'bounding_box': box,
    }
