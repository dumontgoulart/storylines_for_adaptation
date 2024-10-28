'''
Three strategies are considered:

1) Hold the line

2) Retreat & Protect

3) Warning & evacuation

The first strategy, Hold the line, is implemented by building a dike around the entire coast of Beira. 

The second strategy, Retreat & Protect, is implemented by a combination of measures:
    a) Raising the elevation of the port area by 1m
    b) Building dikes along the central part of the coast and then around the inland part of the city (but not on the coast)
    c) Moving the informal settlements to a safer part of the city

The third strategy, Warning & evacuation, is implemented by emulating a system of early warning and evacuation, in which 
the number of people exposed to flooding is reduced by XX%, but no assets are moved.
    
'''

# Measures

# Hold the line
long_sea_wall = {
    "name": "long_sea_wall",
    "description": "long_sea_wall",
    "type": "floodwall",
    "polygon_file": "beira_seawall.geojson",
    "selection_type": "polyline",
    "elevation": {
        "value": 4.0,
        "units": "meter"
    }
}

# Retreat & Protect
raise_port = {
    "name": "raise_port_polygon",
    "description": "raise_port_polygon",
    "type": "elevate_properties",
    "polygon_file": "beira_port.geojson",
    "elevation": {
        "value": 1,
        "units": "m",
        "type": "floodmap"
    },
    "selection_type": "polygon",
    "property_type": "ALL"
}

retreat_sea_wall = {
    "name": "retreat_sea_wall",
    "description": "retreat_sea_wall",
    "type": "floodwall",
    "polygon_file": "beira_internal_seawall.geojson",
    "selection_type": "polyline",
    "elevation": {
        "value": 2.0,
        "units": "meter"
    }
}

buyout_informal = {
    "name": "buyout_informal",
    "description": "buyout_informal",
    "type": "buyout_properties",
    "polygon_file": "beira_informal_settlement.geojson",
    "selection_type": "polygon"
}

# Warning & evacuation
#TODO: this is not implemented yet, but we can do something similar to buyout or elevate and just change the exposure on number of people
evacuation = {
    "name": "evacuation",
    "description": "evacuation",
    "type": "evacuation",
    "selection_type": "ALL",
    "evacuation_efficacy": {
        "value": 40, # to be determined later
        "units": "percentage"
    }
}

############
# Define the strategies
############
hold_the_line = {
    "name": "hold_the_line",
    "description": "hold_the_line",
    "measures": ["long_sea_wall"]
}

retreat_and_protect = {
    "name": "retreat_and_protect",
    "description": "retreat_and_protect",
    "measures": ["raise_port", "retreat_sea_wall", "buyout_informal"]
}

warning_and_evacuation = {
    "name": "warning_and_evacuation",
    "description": "warning_and_evacuation",
    "measures": ["evacuation"]
}
############