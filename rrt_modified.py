# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:30:54 2020
111111
@author: prana
"""
import numpy as np
from random import seed
from random import randint
import matplotlib.pyplot as plt
import time
import cv2


class Node():
    def __init__(self,co_ord):
        self.co_ord=co_ord
        self.neighbours=[]
        self.obstacle=None
        self.cost=0
        self.cost_togo=0
def findDistance(p1,p2):
    return np.sqrt(np.square(p1.co_ord[0]-p2.co_ord[0])+np.square(p1.co_ord[1]-p2.co_ord[1]))


def Maze():
    Maze=[]
    for i in range(501):
        temp=[]
        for j in range(501):
            temp.append(Node([501,501]))
        Maze.append(temp)
        
    for i in range(501):
        for j in range(501):
            
            n=Node([i,j])
            n.obstacle=0
            Maze[i][j]=n
            if i>=50   and i<=450 and j>=50  and j<=450:
                Maze[i][j].obstacle=1 #change to 1
    return Maze
            
            

def closestPoint(point_,Nodes):
    d_smallest=float('inf')
    point_smallest_=Node([500,500])
    for node in Nodes:
        #print(point_)
        #print(node)
        #print(" ")
        d=findDistance(point_,node)
        #print(d)
        if d<d_smallest:
            d_smallest=d
            point_smallest_=node
        #print([d_smallest,point_smallest_.co_ord])
    return point_smallest_




def RRT(start,counter_,delta):  
    
    Maze_=Maze()
    Nodes=[]
    start_=Maze_[start[0]][start[1]]
    goal=[]
    goal_node_=Maze_[475][475]
    for i in range(450,500):                           #assign goal nodes   make this one node and try to have a func to check if goal or not
        temp=[]
        for j in range(450,500):
            temp.append(Maze_[i][j])
        goal.append(temp)
        
    Nodes.append(start_)
    counter=0
    flag=0
    start_.cost=0
    while 1:#counter<counter_:
        counter=counter+1 
        Nodes_list=[]
        for i in range(1):
            x=randint(0,500)
            y=randint(0,500)
            point_=Maze_[x][y]
            point_neighbour_=closestPoint(point_,Nodes)
            if Maze_[x][y].obstacle==1:
                continue
            #write heurestic part here
            
            #interpolate function
            theta=np.arctan2(point_.co_ord[1]-point_neighbour_.co_ord[1],point_.co_ord[0]-point_neighbour_.co_ord[0])
            new_x=x
            new_y=y
            if findDistance(point_,point_neighbour_)>delta:
                
                new_x=point_neighbour_.co_ord[0]+(delta*np.cos(theta))
                new_y=point_neighbour_.co_ord[1]+(delta*np.sin(theta))
                #if new_x<x    or  new_y<y:
                    #continue
                if new_x>500:
                    new_x=495
                if new_x<0:
                    new_x=0
                if new_y>500:
                    new_y=495
                if new_y<0:
                    new_y=0
                point_=Maze_[int(new_x)][int(new_y)]
            
            
            if Maze_[int(new_x)][int(new_y)].obstacle==1:
                counter=counter-1
                continue
            #print([new_x,new_y])
            
            if pathinObstacle(point_,point_neighbour_,Maze_)==True:
                counter=counter-1
                #print("hi")
                #print([point_neighbour_.co_ord,point_.co_ord])
                continue
            
            point_.cost=point_neighbour_.cost+euler_dist(point_,point_neighbour_)#+euler_dist(point_,goal_node_)
            Nodes_list.append([point_,point_neighbour_])
        
        if len(Nodes_list)==0:
            #print("Node list 0")
            continue
        smallest_cost=100000000
        smallest_cost_node_=Maze_[50][50]
        for p in Nodes_list:
            if p[0].cost<smallest_cost:
                smallest_cost_node_=p[0]
                smallest_cost=p[0].cost
                point_neighbour_=p[1]
                
            
        point_=smallest_cost_node_
        point_neighbour_.neighbours.append(point_)
        print([point_neighbour_.co_ord,point_.co_ord])
        #plt.axis([0, 500, 0, 500])
        #plotLine(point_.co_ord[0],point_.co_ord[1],point_neighbour_.co_ord[0],point_neighbour_.co_ord[1],'yo')
        #plt.pause(0.05)
        #printPointneighbours(point_neighbour_)#,point_neighbour_.neighbours)
        Nodes.append(point_)
        #plt.plot(point_.co_ord[0],point_.co_ord[1],'ro')
        #time.sleep(2)
        
        #is goal func
        if point_.co_ord[0]>450   and  point_.co_ord[0]<500  and point_.co_ord[1]>450   and  point_.co_ord[1]<500:
            flag=1
            print(Nodes[len(Nodes)-1].co_ord)
            break
        
    
    if flag==1:
        print("Maze Solved")
        print("Iterations= ")
        print(counter)
    else:
        print("Could not solve")
    
    X=[]
    Y=[]
    Xdash=[]
    Ydash=[]
    Xg=[]
    Yg=[]
    for i in range(len(Nodes)):
        X.append(Nodes[i].co_ord[0])
        Y.append(Nodes[i].co_ord[1])
    
        
    for i in range(50,450):
        for j in range(50,450):
            Xdash.append(i)
            Ydash.append(j)
            
    for i in range(450,500):
        for j in range(450,500):
            Xg.append(i)
            Yg.append(j)
    
    #print(path_)
    Xpath=[]
    Ypath=[]
    
 
    
    plt.plot(Xpath,Ypath,'mo')
    plt.plot(X,Y, 'ro',markersize=1)
    
    plt.plot(Xdash,Ydash,'bo')
    plt.plot(Xg,Yg,'yo')
    #plotLine(0,0,300,200)
    
    
    print(Nodes[0].co_ord)             #Start is start
    print(Nodes[len(Nodes)-1].co_ord)    #last is goal
    path_=maze_solver_bfs(Nodes,Nodes[len(Nodes)-1])   
    
    plt.show()
#    for n in range (1,len(Nodes)):
#        
#        for n1 in Nodes[n].neighbours:
#            #print(n1.co_ord)
            


#pass the last node and it will go to the first node
def maze_solver_bfs(Nodes,node):
    print("Sol of BFS")
    
    
    

    path=[]
    
    print("loop started")
    
    start=node
    path=[]
    path.append(start)
    while 1:
        start=SearchinNeighbour(start,Nodes)
        path.append(start)
        #print(start.co_ord)
        if start==Nodes[0]:       #i.e.start point
            break
    print("path is")
    for c in path:
        print(c.cost)
    for i in range(len(path)-1):
        plotLine(path[i].co_ord[0],path[i].co_ord[1],path[i+1].co_ord[0],path[i+1].co_ord[1],'ro')
        
        
        
def SearchinNeighbour(start,Nodes):
    for c in Nodes:
        for d in c.neighbours:
            if d==start:
                return c
    


def euler_dist(node1_,node2_):
    x1,y1=node1_.co_ord
    x2,y2=node2_.co_ord
    return (((y2-y1)**2)+((x2-x1)**2))**(0.5)



        
        
    
    
   
def findIndex(n,Nodes):
    for i in range(len(Nodes)):
        if Nodes[i]==n:
            return i
    return -1
def presentIndex(n,Nodes):
    for node in Nodes:
        if node==n:
            return True
    return False

def plotLine(x1,y1,x2,y2,color):#,Xpath,Ypath):
    if x1>x2   and x1-x2!=0:
        [x1,y1,x2,y2]=[x2,y2,x1,y1]
    if x1!=x2:    
        for d in np.arange(x1,x2,0.25).tolist():
            if x2-x1!=0:
                y=(y2-y1)*(d-x1)/(x2-x1)+y1
    
                plt.plot(d,y,color,markersize=2)
        
    if x1==x2:
        print("hi")
        if y1>y2:
            [x1,y1,x2,y2]=[x2,y2,x1,y1]
        for d in np.arange(y1,y2,0.25).tolist():
            plt.plot(x1,d,color,markersize=2)
        #print("hi")
    #time.sleep(0.05)
        #Xpath.append(d)
        #Ypath.append(y)
        #print(Xpath)
        #print(Ypath)
        #return [[Xpath],[Ypath]]
    
def printPointneighbours(point_neighbour_):
    lists=[]
    for c in point_neighbour_.neighbours:
        lists.append(c.co_ord)
    print([point_neighbour_.co_ord,lists])
    
def pathinObstacle(point_,point_neighbour_,Maze_):
    [x1,y1]=point_.co_ord
    [x2,y2]=point_neighbour_.co_ord
    
    if x1>x2   and x1-x2!=0:
        [x1,y1,x2,y2]=[x2,y2,x1,y1]
        
    for d in np.arange(x1,x2).tolist():
        if x2-x1!=0:
            y=(y2-y1)*(d-x1)/(x2-x1)+y1
            
        else:
            y=y1
        p=Maze_[int(d)][int(y)]
        if p.obstacle==1:
            return True
    return False

RRT([0,0],1000,50)


    
    