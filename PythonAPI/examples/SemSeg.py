#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
from os import listdir
from os.path import isfile, join

#Keywords for Semantic Labelling
Building = ["wall","Parkwall", "SecFence",  "Windmill", "Building", "House", "Terraced", "Skyscraper", "Office", "Mansion", "Ladder", "GasStation", "Garage", "farmHouse", "Church", "Block", "Apartment", "AirConditioner", "Stairs"]
Fence = ["Fence", "Barrier", "Fences", "Guard", "SecWaterDrums", "Guardrail", "TreeBark"]
Pole = ["Freewaylight","Railtrain", "Railtrack" , "Bollard", "Baserailtrain", "bridgepillar", "Light", "Streetlight", "columntunnel", "electricpole", "TLight" "Firehdrant", "Parklight", "Trafficlight", "Trafficpole", "Addcartel", "InterchangeSign", "Powerpole", "Splinepoweline", "Trafficcones", "cones"]
Road = ["Road", "MarkingNode" "Asphalt", "Concrete", "Grass", "Phong", "LaneMarking", "Lane", "Ramp", "RoadPiece", "IntersectionEntrance", "AlphaPaint" ]
SideWalk = ["curb", "SideWalk", "Pathway", "Closehole", "Square", "ManholeCover"]
Vegetation = ["Tree", "Pot", "CoconutPalm", "Palm", "Veg", "amapola", "walnut", "Billboard", "Acacia", "Pine", "Bush", "Cypress", "Trunk", "Leaf", "Grassleaf", "Platanus", "Leaves", "Hedge", "Acer", "Saccharum", "DatePalm", "Mexican_Fan_Palm", "Arbusto", "Cherry_Bark", "Fir_Bark ", "Japanese_Maple", "Pine_leaf", "Platanus_Ash", "Quercus_Rubra", "Sassafras", "Willow", "Twig", "Branch", "Scots_pine_trunk", "White_ASh_Trunk", "PlantCorn", "TeethOfthelion", " WheatField", "Rushes", "Beech"]
TrafficSigns = ["AnimalCrossing", "LaneReduc", "Left", "Stop", "SpeedLimit", "OneWay", "Yield", "MasterSigns", "AnimaLCrossing", "WildCrossing", "NoTurn", "DoNotEnter", "InterchangeSign", "RoundSign"  ]
Prop = ["Prop"]
Vehicles = ["CrossBike", "Harley", "Kawasakininja", "LeisureBike", "RoadBike", "Vespa", "Yamaha", "4WheeledBike", "PedestrianOnsidecar", "Audi", "Beetle", "Bmw", "CarlaCola", "Chevrolet", "Citroen", "DodgeCharge", "Jeep", "Leon", "Lincoln", "Mercedes", "Mini", "Mustang", "Nissan", "Tazzari", "Tesla", "Toyota", "Truck", "Volkswagen", "Wheeled", "Van" ] 
Cont_entity = ["RepSpline", "Spline"]

mypath = "C:/test/plc_labels/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [mypath + f for f in onlyfiles]
onlyfiles

try:
    os.makedirs("C:/test/plc_labels/SemSeg", exist_ok=True)
except FileExistsError:
    # directory already exists
    pass

for y in onlyfiles:
    actuallist = pd.read_csv(y)
    for x in Vegetation:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Vegetation'
    for x in Building:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Building'
    for x in Prop:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Prop'
    for x in Vehicles:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Vehicles'
    for x in Fence:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Fence'
    for x in Pole:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Pole'
    for x in Road:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Road'
    for x in SideWalk:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'SideWalk'
    for x in TrafficSigns:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'TrafficSigns'
    for x in Cont_entity:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = 'Continuous Wall/Barricade'
    actuallist.to_csv('C:/test/plc_labels/SemSeg/{}'.format(y[19:]), header=True, index=False)

    print("Semantic Labels transformed for file {}".format(y[19:]))







