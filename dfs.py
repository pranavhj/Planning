# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:20:36 2020

@author: prana
"""
import numpy as np
import time
def dist(current,parent):
    dist=np.sqrt(np.square(current[0]-parent[0])+np.square(current[1]-parent[1]))
    return dist

def createMaze():
    maze = []
    maze.append(["#","#", "#", "#", "#", "O","#"])
    maze.append(["#"," ", " ", " ", "#", " ","#"])
    maze.append(["#"," ", "#", " ", "#", " ","#"])
    maze.append(["#"," ", "#", " ", " ", " ","#"])
    maze.append(["#"," ", "#", "#", "#", " ","#"])
    maze.append(["#"," ", " ", " ", "#", " ","#"])
    maze.append(["#","#", "#", "#", "#", "X","#"])

    return maze

def createMaze2():
    maze = []
    maze.append(["#","#", "#", "#", "#", " ", "#", "#", "#"])
    maze.append(["#"," ", " ", " ", " ", " ", " ", " ", "#"])
    maze.append(["#"," ", "#", "#", " ", "#", "#", " ", "#"])
    maze.append(["#"," ", "#", " ", " ", " ", "#", " ", "#"])
    maze.append(["#"," ", "#", " ", "#", " ", "#", " ", "#"])
    maze.append(["#"," ", "#", " ", "#", " ", "#", "O", "#"])
    maze.append(["#"," ", "#", " ", "#", " ", "#", "#", "#"])
    maze.append(["#"," ", " ", " ", " ", " ", " ", " ", "#"])
    maze.append(["#","#", "#", "#", "#", "#", "#", "X", "#"])

    return maze

def maze_solve_dfs():
    print("Sol of DFS")
    maze=createMaze2()
    start=[]
    goal=[]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=="O" :
                start=[i,j]
            if maze[i][j]=="X" :
                goal=[i,j]
    maze_size=[len(maze),len(maze[0])]
    stack=[]
    visited=np.zeros((maze_size))
    #print(visited)
    stack.append(start)
    path=[]
    
    while len(stack)!=0:
        current_cell = stack[len(stack)-1]
        #print(current_cell)
        path.append(current_cell)
        stack.pop(len(stack)-1)
        if current_cell==goal:
            break
        if visited[current_cell[0]][current_cell[1]] == 0 :
            visited[current_cell[0]][current_cell[1]]=1
        if current_cell[0]!=maze_size[0]-1  and   maze[current_cell[0]+1][current_cell[1]]!="#"  and  visited[current_cell[0]+1][current_cell[1]]==0 :
            stack.append([current_cell[0]+1,current_cell[1]])     #go down
            
        if current_cell[1]!=maze_size[1]-1  and   maze[current_cell[0]][current_cell[1]+1]!="#"  and  visited[current_cell[0]][current_cell[1]+1]==0 :
            stack.append([current_cell[0],current_cell[1]+1])    #go right 
        
        if current_cell[0]!=0  and   maze[current_cell[0]-1][current_cell[1]]!="#"  and  visited[current_cell[0]-1][current_cell[1]]==0 :
            stack.append([current_cell[0]-1,current_cell[1]])    #go up
        
        if current_cell[1]!=0  and   maze[current_cell[0]][current_cell[1]-1]!="#"  and  visited[current_cell[0]][current_cell[1]-1]==0 :
            stack.append([current_cell[0],current_cell[1]-1])    #go left
    if len(stack)==0:    
        print("No sol")
    else:
        for i in path:
            #print(i)
            maze[i[0]][i[1]]="="
        maze[start[0]][start[1]]="O"
        maze[goal[0]][goal[1]]="X"
        print(maze)
        
    
            





def maze_solve_bfs():
    print("Sol of BFS")
    maze=createMaze2()
    start=[]
    goal=[]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=="O" :
                start=[i,j]
            if maze[i][j]=="X" :
                goal=[i,j]
    maze_size=[len(maze),len(maze[0])]
    queue=[]
    path=[]
    visited=np.zeros((maze_size))
    
    #examined=np.zeros((maze_size))
    #print(visited)
    queue.append(start)
    while len(queue)!=0:
        current_cell = queue[len(queue)-1]
        #print("current cell")
        #print(current_cell)
        path.append(current_cell)
        queue.pop(0)
        visited[current_cell[0]][current_cell[1]]=1
        if current_cell==goal:
            break
        neighbours=[]
        if current_cell[0]!=maze_size[0]-1  and   maze[current_cell[0]+1][current_cell[1]]!="#"  and  visited[current_cell[0]+1][current_cell[1]]==0 :
            neighbours.append([current_cell[0]+1,current_cell[1]])     #go down
        
        if current_cell[1]!=maze_size[1]-1  and   maze[current_cell[0]][current_cell[1]+1]!="#"  and  visited[current_cell[0]][current_cell[1]+1]==0 :
            neighbours.append([current_cell[0],current_cell[1]+1])    #go right 
        
        if current_cell[0]!=0  and   maze[current_cell[0]-1][current_cell[1]]!="#"  and  visited[current_cell[0]-1][current_cell[1]]==0 :
            neighbours.append([current_cell[0]-1,current_cell[1]])    #go up
        
        if current_cell[1]!=0  and   maze[current_cell[0]][current_cell[1]-1]!="#"  and  visited[current_cell[0]][current_cell[1]-1]==0 :
            neighbours.append([current_cell[0],current_cell[1]-1])    #go left
        #print(neighbours)
        for c in neighbours:
            if c==goal :
                break
            #print(c)
            queue.append(c)
            visited[c[0]][c[1]]=1
    #print(path)
    for i in path:
        maze[i[0]][i[1]]="="
    maze[start[0]][start[1]]="O"
    maze[goal[0]][goal[1]]="X"
    print(maze)



def maze_solver_dijkstra():
    print("Sol of dijkstra")
    maze=createMaze2()
    zeros=[0,0]
    start=[]
    goal=[]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=="O" :
                start=[i,j]
            if maze[i][j]=="X" :
                goal=[i,j]
    maze_size=[len(maze),len(maze[0])]
    my_maze=np.zeros((maze_size))
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=="O" :
                my_maze[i][j]=5
            if maze[i][j]=="X" :
                my_maze[i][j]=6                 #mark goal start and obstacles
            if maze[i][j]==" ":
                my_maze[i][j]=1
            if maze[i][j]=="#":
                my_maze[i][j]=2
    distance_from_start=np.zeros((maze_size))
    parent=np.zeros(([maze_size[0],maze_size[1],2]))
    
    for i in range(len(distance_from_start)):
        for j in range(len(distance_from_start[0])):
            distance_from_start[i][j]=float('inf');
    numExpanded=0
    distance_from_start[start[0]][start[1]]=0
    
    
    current=start
    
    
    while 1:
        my_maze[start[0]][start[1]]=5
        my_maze[goal[0]][goal[1]]=6
        
        min_dist=float('inf')
        for i in range(len(distance_from_start)):
            for j in range(len(distance_from_start[0])):
                
                if min_dist>distance_from_start[i][j]:                  #stepping forward from the same min distance valued node in every iteration
                    #print([i,j])
                    min_dist=distance_from_start[i][j]
                    current[0]=i
                    current[1]=j
                    
                
        
        if current==goal  or min_dist==float('inf'):
            break
        my_maze[current[0]][current[1]]=3                           #node is visited
        distance_from_start[current[0]][current[1]]=float('inf')     #so that next node is explored
        
        if current[1]<maze_size[1]:
            right1=[current[0],current[1]+1]
        else:
            right1=current                                  #right
            
        
        if current[1]>0:
            left1=[current[0],current[1]-1]
        else:                                                  #left
            left1=current                
            
            
        if current[0]<maze_size[0]:
            down1=[current[0]+1,current[1]]
        else:                                                   #down                       
            down1=current
        

        if current[0]>0:
            up1=[current[0]-1,current[1]]
        else:                                                   #up                       
            up1=current
        if current[0]<maze_size[0]  and  current[1]<maze_size[1]  :
            downright=[current[0]+1,current[1]+1]     # down   right
        else:                                                                          
            downright=current
            
        
        if current[0]<maze_size[0]  and  current[1]>0 :
            downleft=[current[0]+1,current[1]-1]     # down   left
        else:                                                                          
            downleft=current


        if current[0]>0  and  current[1]>0 :
            upleft=[current[0]-1,current[1]-1]     # up   left
        else:                                                                          
            upleft=current
            
            
        
        if current[0]>0  and  current[1]<maze_size[1] :
            upright=[current[0]-1,current[1]+1]     # up   right
        else:                                                                          
            upright=current


        adjacent=[]
        adjacent.append(up1)
        adjacent.append(down1)
        adjacent.append(left1)
        adjacent.append(right1)
        adjacent.append(upleft)
        adjacent.append(upright)
        adjacent.append(downleft)
        adjacent.append(downright)
        temp=0
        
        for n in range(len(adjacent)):
            #Node is either unvisited/empty     or      looked at     or     goal     that is should not be visited or should not be a obstacle
            if my_maze [int(adjacent[n][0])] [int(adjacent[n][1])] ==1    or  my_maze[adjacent[n][0]][adjacent[n][1]]==4   or  my_maze[adjacent[n][0]][adjacent[n][1]]==6 :
                if  distance_from_start[int(adjacent[n][0])] [int(adjacent[n][1])] > min_dist+dist(current,adjacent[n]):
                    distance_from_start[int(adjacent[n][0])] [int(adjacent[n][1])]=min_dist+dist(current,adjacent[n])
                    
                    parent[int(adjacent[n][0])] [int(adjacent[n][1])]=current
                    temp=n                #so that we know the node entered this loop
                my_maze[int(adjacent[n][0])] [int(adjacent[n][1])]=4                               #node is looked at
        numExpanded=numExpanded+1
        current=adjacent[temp]
        #print(distance_from_start)
        #time.sleep(1)
        #print(parent)
    route=[]
    if distance_from_start[goal[0]][goal[1]]==float('inf'):
        route=[]
    else:
        route=[goal]
        #print(([route[len(route)-1][0]][route[len(route)-1][1]][0]))
        #print(int(parent[route[len(route)-1][0]][route[len(route)-1][1]][1]))
        a=route[len(route)-1][0]
        b=route[len(route)-1][1]
        #print(parent[a][b][0])
        while parent[int(a)][int(b)][0]!=0   and  parent[int(a)][int(b)][1]!=0 :
            a=route[len(route)-1][0]
            b=route[len(route)-1][1]
            c1=parent[int(a)][int(b)][0]
            c2=parent[int(a)][int(b)][1]
            route.append([c1,c2])

            
    #print(route)
    for i in route:
        maze[int(i[0])][int(i[1])]="="
    maze[start[0]][start[1]]="O"
    maze[goal[0]][goal[1]]="X"
    print(maze)
    
 


def maze_solver_Astar():
    print("Sol of Astar")
    maze=createMaze2()
    zeros=[0,0]
    start=[]
    goal=[]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=="O" :
                start=[i,j]
            if maze[i][j]=="X" :
                goal=[i,j]
    maze_size=[len(maze),len(maze[0])]
    my_maze=np.zeros((maze_size))
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]=="O" :
                my_maze[i][j]=5
            if maze[i][j]=="X" :
                my_maze[i][j]=6                 #mark goal start and obstacles
            if maze[i][j]==" ":
                my_maze[i][j]=1
            if maze[i][j]=="#":
                my_maze[i][j]=2
    
    parent=np.zeros(([maze_size[0],maze_size[1],2]))
    X=np.zeros((maze_size[0]))
    Y=np.zeros((maze_size[1]))
    H=np.zeros((maze_size[0],maze_size[1]))
    for i in range(maze_size[0]):
        for j in range(maze_size[1]):
            [X[i],Y[i]]=[i,j]
            H[i][j]=abs(X[i]-goal[0])+abs(Y[i]-goal[1])
    g=np.zeros((maze_size[0],maze_size[1]))
    f=np.zeros((maze_size[0],maze_size[1]))
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            g[i][j]=float('inf')
            f[i][j]=float('inf')
    g[start[0]][start[1]]=0
    f[start[0]][start[1]]=H[start[0]][start[1]]
    numExpanded=0
    
    
    
    current=start
    
    
    while 1:
        my_maze[start[0]][start[1]]=5
        my_maze[goal[0]][goal[1]]=6
        
        min_dist=float('inf')
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                
                if min_dist>f[i][j]:
                    #print([i,j])
                    min_dist=f[i][j]
                    current[0]=i
                    current[1]=j
                    
                
        
        if current==goal  or min_dist==float('inf'):
            break
        my_maze[current[0]][current[1]]=3
        f[current[0]][current[1]]=float('inf')
        
        if current[1]<maze_size[1]:
            right1=[current[0],current[1]+1]
        else:
            right1=current                                  #right
            
        
        if current[1]>0:
            left1=[current[0],current[1]-1]
        else:                                                  #left
            left1=current                
            
            
        if current[0]<maze_size[0]:
            down1=[current[0]+1,current[1]]
        else:                                                   #down                       
            down1=current
            
        
        
        if current[0]>0:
            up1=[current[0]-1,current[1]]
        else:                                                   #up                       
            up1=current
            
            
        if current[0]<maze_size[0]  and  current[1]<maze_size[1]  :
            downright=[current[0]+1,current[1]+1]     # down   right
        else:                                                                          
            downright=current
            
        
        if current[0]<maze_size[0]  and  current[1]>0 :
            downleft=[current[0]+1,current[1]-1]     # down   left
        else:                                                                          
            downleft=current


        if current[0]>0  and  current[1]>0 :
            upleft=[current[0]-1,current[1]-1]     # up   left
        else:                                                                          
            upleft=current
            
            
        
        if current[0]>0  and  current[1]<maze_size[1] :
            upright=[current[0]-1,current[1]+1]     # up   right
        else:                                                                          
            upright=current


        adjacent=[]
        adjacent.append(up1)
        adjacent.append(down1)
        adjacent.append(left1)
        adjacent.append(right1)
        adjacent.append(upleft)
        adjacent.append(upright)
        adjacent.append(downleft)
        adjacent.append(downright)
        temp=0
        
        for n in range(len(adjacent)):
            
            if my_maze [int(adjacent[n][0])] [int(adjacent[n][1])] ==1    or  my_maze[adjacent[n][0]][adjacent[n][1]]==4   or  my_maze[adjacent[n][0]][adjacent[n][1]]==6 :
                if  g[int(adjacent[n][0])] [int(adjacent[n][1])] > g[current[0]][current[1]]+1:
                    g[int(adjacent[n][0])] [int(adjacent[n][1])]=g[current[0]][current[1]]+1
                    f[int(adjacent[n][0])] [int(adjacent[n][1])]=g[int(adjacent[n][0])] [int(adjacent[n][1])]+H[int(adjacent[n][0])] [int(adjacent[n][1])]
                    
                    parent[int(adjacent[n][0])] [int(adjacent[n][1])]=current
                    temp=n
                my_maze[int(adjacent[n][0])] [int(adjacent[n][1])]=4
        numExpanded=numExpanded+1
        print(adjacent)
        current=adjacent[temp]
        
        #print(parent)
    route=[]
    if f[goal[0]][goal[1]]==float('inf'):
        route=[]
    else:
        route=[goal]
        #print(([route[len(route)-1][0]][route[len(route)-1][1]][0]))
        #print(int(parent[route[len(route)-1][0]][route[len(route)-1][1]][1]))
        a=route[len(route)-1][0]
        b=route[len(route)-1][1]
        #print(parent[a][b][0])
        while parent[int(a)][int(b)][0]!=0   or  parent[int(a)][int(b)][1]!=0 :
            a=route[len(route)-1][0]
            b=route[len(route)-1][1]
            c1=parent[int(a)][int(b)][0]
            c2=parent[int(a)][int(b)][1]
            route.append([c1,c2])

            
    #print(route)
    for i in route:
        maze[int(i[0])][int(i[1])]="="
    maze[start[0]][start[1]]="O"
    maze[goal[0]][goal[1]]="X"
    print(maze)
    


            
    
    

def ind2sub(array_shape, ind):
    rows=(ind.astype('int')/array_shape[1])
    cols=(ind.astype('int')%array_shape[1])
    return (rows,cols)

def sub2ind(array_shape,rows,cols):
    ind=rows*array_shape[1]+cols
    #ind[ind<0]=-1
    #ind[ind>=array_shape[0]*array_shape[1]]=-1
    return ind


#def maze_solve_dijkstra():
#    print("Sol of dijkstra")
#    print("Sol of BFS")
#    maze=createMaze2()
#    start=[]
#    goal=[]
#    for i in range(len(maze)):
#        for j in range(len(maze[0])):
#            if maze[i][j]=="O" :
#                start=[i,j]
#            if maze[i][j]=="X" :
#                goal=[i,j]
#    maze_size=[len(maze),len(maze[0])]
#    queue=[]
#    path=[]
#    visited=np.zeros((maze_size))
#    distances=np.zeros((maze_size))
#    for i in range(len(distances)):
#        for j in range(len(distances[0])):
#            distances[i][j]=999999999;
#    current_cell=start
#    distances[current_cell[0]][current_cell[1]]=0
#    while current_cell!=goal:    
#        
#        
#        neighbours=[]
#    
#        if current_cell[0]!=maze_size[0]-1  and   maze[current_cell[0]+1][current_cell[1]]!="#"  and  visited[current_cell[0]+1][current_cell[1]]==0 :
#                neighbours.append([current_cell[0]+1,current_cell[1]])     #go down
#            
#        if current_cell[1]!=maze_size[1]-1  and   maze[current_cell[0]][current_cell[1]+1]!="#"  and  visited[current_cell[0]][current_cell[1]+1]==0 :
#            neighbours.append([current_cell[0],current_cell[1]+1])    #go right 
#        
#        if current_cell[0]!=0  and   maze[current_cell[0]-1][current_cell[1]]!="#"  and  visited[current_cell[0]-1][current_cell[1]]==0 :
#            neighbours.append([current_cell[0]-1,current_cell[1]])    #go up
#        
#        if current_cell[1]!=0  and   maze[current_cell[0]][current_cell[1]-1]!="#"  and  visited[current_cell[0]][current_cell[1]-1]==0 :
#            neighbours.append([current_cell[0],current_cell[1]-1])    #go left
#    
#        
#        
#        
#        if current_cell[0]!=maze_size[0]-1  and  current_cell[1]!=maze_size[1]-1  and   maze[current_cell[0]+1][current_cell[1]+1]!="#"  and  visited[current_cell[0]+1][current_cell[1]+1]==0 :
#                neighbours.append([current_cell[0]+1,current_cell[1]+1])     # down   right
#            
#        if current_cell[0]!=maze_size[0]-1  and  current_cell[1]!=0  and   maze[current_cell[0]+1][current_cell[1]-1]!="#"  and  visited[current_cell[0]+1][current_cell[1]-1]==0 :
#                neighbours.append([current_cell[0]+1,current_cell[1]-1])     # down   left
#        
#        if current_cell[0]!=0  and  current_cell[1]!=maze_size[1]-1  and   maze[current_cell[0]-1][current_cell[1]+1]!="#"  and  visited[current_cell[0]-1][current_cell[1]+1]==0 :
#                neighbours.append([current_cell[0]-1,current_cell[1]+1])     # up   right
#        
#        if current_cell[0]!=0  and  current_cell[1]!=0  and   maze[current_cell[0]-1][current_cell[1]-1]!="#"  and  visited[current_cell[0]-1][current_cell[1]-1]==0 :
#                neighbours.append([current_cell[0]-1,current_cell[1]-1])     # up   left
#            
#        dist_array=[] 
#        cell_array=[]
#        for c in neighbours:
#            #print(c)
#            dist=np.sqrt(np.square(start[0]-c[0])+np.square(start[1]-c[1]))
#           
#            
#            if dist<distances[c[0]][c[1]]:
#                distances[c[0]][c[1]]=dist
#            dist_array.append(distances[c[0]][c[1]])
#            cell_array.append(c)
#        visited[current_cell[0]][current_cell[1]]=1
#        print(dist_array)
#        if(len(dist_array)==0):
#            break
#        min_dist=min(dist_array);
#        min_dist_index=dist_array.index(min_dist)
#        current_cell=cell_array[min_dist_index]
#        print(current_cell)
#        path.append(current_cell)
#    for i in path:
#        maze[i[0]][i[1]]="="
#    print(maze)
            

#maze_solve_dfs()
#maze_solve_bfs()               
maze_solver_dijkstra()   
#maze_solver_Astar() 
    
