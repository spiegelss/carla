import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import datetime
import glob

#Keywords for mapping assets and Target Classes

Car = ["PedestrianOnsidecar", "Audi", "Beetle", "Bmw",  "Chevrolet", "Citroen", "DodgeCharge", "Jeep", "Leon", "Lincoln", "Mercedes", "Mini", "Mustang", "Nissan", "Tazzari", "Tesla", "Toyota", "Volkswagen", "Wheeled", "Van" ]
Motorbicycle = [ "Harley", "Kawasakininja",  "Vespa", "Yamaha", "4WheeledBike"]
Bicycle = ["CrossBike","LeisureBike", "RoadBike"]
Truck = ["Truck", "CarlaCola" ]
Road = ["Floor","Road", "MarkingNode" "Asphalt", "Concrete", "Phong", "LaneMarking", "Lane", "Ramp", "RoadPiece", "IntersectionEntrance", "AlphaPaint" ]
SideWalk = ["SideWalkCube","curb", "SideWalk", "Pathway", "Closehole", "Square", "ManholeCover"]
Fence = ["Wall", "Parkwall", "Fence", "Barrier", "Fences", "Guard", "SecWaterDrums", "Guardrail", "TreeBark"]
TrafficLight = ["Tlight", "Trafficlight"]
TrafficCone = ["Trafficcones", "cones"]
Pole = ["pole", "Freewaylight","Railtrain", "Railtrack" , "Bollard", "Baserailtrain", "bridgepillar", "Light", "Streetlight", "columntunnel", "electricpole", "Firehdrant", "Prop_Parklight", "Trafficpole", "Addcartel", "InterchangeSign", "Powerpole", "Splinepoweline"]
TrafficSigns = ["AnimalCrossing", "LaneReduc", "Left", "Stop", "SpeedLimit", "OneWay", "Yield", "MasterSigns", "AnimaLCrossing", "WildCrossing", "NoTurn", "DoNotEnter", "InterchangeSign", "RoundSign"  ]
Prop = ["Prop", "MapTable", "LargePlantPot", "plant_pit", "Wire", "HighvoltageCable", "Awning", "Clothesline", "ConstructionCone", "StreetBarrier", "HayBale", "Hay", "Barbecue", "DogHouse", "Fountain", "GardenLamp", "Gnome", "Pergola", "PlasticChair", "PlasticTable", "SwingCouch", "Table", "trampoline", "Umbrella", "Table", "GuardShelter", "IceFreezer", "Slide", "ParkingBarrier", "Bikehelmet", "Guitarcase", "briefcase" , "PlasticBag", "Purse", "ShoppinCart", "Trolley", "Travelcase", "Mobile", "Advertise", "ATM", "Bench", "BikeParking", "BusStop", "ChainBarrier", "Letter", "MailBox", "MapTable", "PlatformGarbage", "Garbage", "StandNews", "StreetCounter", "Advertise", "BusStopGlass", "letter", "SM_StreetAD", "trafficcones", "barrel", "bin", "Bigcontainer" , "box", "brokentile", "Clothcontainer", "ColaCan", "container", "Creasedbox", "ColaMachine", "DirtDebris", "ironplank", "Trasdh", "Trash", "BinBody", "BinWheel", "WateringCan"]
Building = ["Hall","NewBluePrint","BP_CarlaCola","Kiosk", "Shop","Mall","wall","Parkwall", "SecFence",  "Windmill", "Building", "House", "Terraced", "Skycraper","Skyscraper", "Office", "Mansion", "Ladder", "GasStation", "Garage", "farmHouse", "Church", "Block", "Apartment", "AirConditioner", "Stairs"]
Vegetation = ["Grass", "Landscape","Maple", "Plant","Tree", "Pot", "CoconutPalm", "Palm", "Veg", "amapola", "walnut", "Billboard", "Acacia", "Pine", "Bush", "Cypress", "Trunk", "Leaf", "Grassleaf", "Platanus", "Leaves", "Hedge", "Acer", "Saccharum", "DatePalm", "Mexican_Fan_Palm", "Arbusto", "Cherry_Bark", "Fir_Bark ", "Japanese_Maple", "Pine_leaf", "Platanus_Ash", "Quercus_Rubra", "Sassafras", "Willow", "Twig", "Branch", "Scots_pine_trunk", "White_ASh_Trunk", "PlantCorn", "TeethOfthelion", " WheatField", "Rushes", "Beech"]
Environment = ["Generic_stone", "Rock"]

#Path for Directory of lidar data

onlyfiles = []
mypath = "D:/Carla_Lidar/2019-06-14 15-52-14/"
os.chdir(mypath)
for file in glob.glob("*.csv"):
    onlyfiles.append(mypath + file)
onlyfiles

for y in onlyfiles:
    actuallist = pd.read_csv(y)
    actuallist.loc[actuallist['labels'].str.contains("Fetch Failed", case=False), 'labels'] = '255'
    for x in Car:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '1'
    for x in Motorbicycle:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '2'
    for x in Bicycle:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '3'
    for x in Truck:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '6'
    for x in Road:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '9'
    for x in SideWalk:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '10'
    for x in Fence:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '13'
    for x in TrafficLight:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '14'
    for x in TrafficCone:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '11'
    for x in Pole:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '15'
    for x in TrafficSigns:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '16'
    for x in TrafficSigns:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '16'
    for x in Prop:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '18'
    for x in Building:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '20'
    for x in Vegetation:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '21'
    for x in Environment:
        actuallist.loc[actuallist['labels'].str.contains(x, case=False), 'labels'] = '255'

    pointcloud = actuallist[['x', 'y', 'z']].apply(pd.to_numeric, downcast='float')
    labels = actuallist['labels'].tolist()
    labels = pd.to_numeric(labels, downcast='unsigned')
    ego_trans = actuallist['ego_X,Y,Z']
    ego_rot = actuallist['ego_rot_P,Y,R']
    Lidar_Params = actuallist['Lidar_C,R,P,F,uF,lF']
    splice_length = len(mypath)
    with h5py.File(os.path.join(mypath, "{}.h5".format(y[splice_length:-4])), 'w') as out_file:
        out_file.create_dataset('data', data=pointcloud.values, dtype=np.float32,
                                chunks=True,
                                compression=1)
        out_file.create_dataset('labels', data=labels, dtype=np.uint8,
                                chunks=True,
                                compression=1)
        out_file.create_dataset('ego_X,Y,Z', data=ego_trans.values, dtype=np.float32,
                                chunks=True,
                                compression=1)
        out_file.create_dataset('ego_rot_P,Y,R', data=ego_rot.values, dtype=np.float32,
                                chunks=True,
                                compression=1)
        out_file.create_dataset('Lidar_C,R,P,F,uF,lF', data=Lidar_Params.values, dtype=np.float32,
                                chunks=True,
                                compression=1)

    print("HDF5 created for the file {}".format(y[splice_length:-4]))
print("Dataset created")