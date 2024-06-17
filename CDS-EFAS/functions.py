# help functions
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_regions(name):
    """Get lat-lon bounds of a region"""
    name = name.lower()
    if name == 'panaro':
        return (43.5,45.5), (10,12)
    if name == 'timis':
        return  (44.5,46.5), (20,23)
    if name == 'lagen':
        return (59.5,61.5), (7, 10.5)
    if name == 'aragon':
        return (41.5,43.5), (-3,1)
    if name == 'global':
        return (24,72), (-35,75)

    raise ValueError(f"Region {name} not found")

def fixed_region(name, delta):
    """Get lat-lon bounds of a region with delta around the mean lat-lon"""
    lat, lon = get_regions(name)
    if name != 'global':
        lat, lon = (lat[0]-delta, lat[1]+delta), (lon[0]-delta, lon[1]+delta)
    return lat, lon

def mask_efas(field, regions):
    """Create a mask for the EFAS reforecast"""
    for region in regions:
        lat, lon = get_regions(region)
        if region == regions[0]:
            mask = ((field.longitude >= lon[0]) & (field.longitude <= lon[1]) & (field.latitude >= lat[0]) & (field.latitude <= lat[1]))
        else:
            mask = mask | ((field.longitude >= lon[0]) & (field.longitude <= lon[1]) & (field.latitude >= lat[0]) & (field.latitude <= lat[1]))

    return field.where(mask)