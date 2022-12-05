import numpy as np

#generators consume elements


def get_distance(s1, s2):
    """_summary_

    Args:
        s1 (_type_): _description_
        s2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)**0.5

# Don't do position, just pass back the list of movements
# Then post process with floor and cap. 
# also could generate a list list of -r to r in each dimension.
# Then use get combinations. -> or rather do it manually as a generator
def get_cells_in_radius(position, radius, ceil = None, flr = [0,0]):
   
    i = max([np.floor(position[0]-radius),flr[0]])
    j = max([np.floor(position[1]-radius),flr[1]])
    if type(ceil) != list:
        max_x = np.ceil(position[0] + radius)
        max_y = np.ceil(position[1] + radius)
    else:
        max_x = min([np.ceil(position[0] + radius), ceil[0]])
        max_y = min([np.ceil(position[1] + radius), ceil[1]])
    dy = max_y - j +1
    
    k = 0
    
    while i + np.floor(k/(dy)) <= max_x:
        d = get_distance(position, [i + np.floor(k/(dy)), j + (k % dy)])
        if d <= radius and d != 0:
            yield [ i + np.floor(k/(dy)), j + (k % dy)]
        k += 1
        
def get_int_in_radius(position, radius, ceil = None, flr = [0,0]):
       
    i = max([np.floor(position[0]-radius),flr[0]])
    j = max([np.floor(position[1]-radius),flr[1]])
    if type(ceil) != list:
        max_x = np.ceil(position[0] + radius)
        max_y = np.ceil(position[1] + radius)
    else:
        max_x = min([np.ceil(position[0] + radius), ceil[0]])
        max_y = min([np.ceil(position[1] + radius), ceil[1]])

    dy = max_y - j +1
    
    k = 0
    
    while i + np.floor(k/(dy)) <= max_x:
        d = get_distance(position, [i + np.floor(k/(dy)), j + (k % dy)])
        if d <= radius and d != 0:
            yield k
        k += 1
    
if __name__=='__main__':
    for el in get_cells_in_radius([2,2], 1, [10,10]):
        print(el) 
    for el in get_int_in_radius([2,2], 1, [10,10]):
        print(el)     

# def get_neighbors(self, _position):
#         neighbors = []
#         neighbors_ind = []
#         step = [[ 0, -1], [-1, -1], [-1,  0], [-1,  1], [ 0,  1], [ 1,  1], [ 1,  0], [ 1, -1]]
#         for i in range(8):
#             t = list(_position)
#             t[0] += step[i][0]
#             t[1] += step[i][1]
            
#             if t[0] >= 0 and t[1] >= 0 and t[0] < self.dim_[0] and t[1] < self.dim_[1]:
#                 neighbors.append(t)
#                 neighbors_ind.append(i)
#         return neighbors, neighbors_ind

#     def act_2_dir(self, _action):
#         if   _action == 0:
#             return [ 0, -1] # S
#         elif _action == 1:
#             return [-1, -1] # SW
#         elif _action == 2:
#             return [-1,  0] #  W
#         elif _action == 3:
#             return [-1,  1] # NW
#         elif _action == 4:
#             return [ 0,  1] # N
#         elif _action == 5:
#             return [ 1,  1] # NE
#         elif _action == 6:
#             return [ 1,  0] #  E
#         elif _action == 7:
#             return [ 1, -1] # SE
#         else:
#             return [0, 0]   #  Z
    
#     def get_coordinate_move(self, _position, _action):
#         _position = _position.copy()
#         step = self.act_2_dir(_action)

#         temp = _position.copy()
#         # print(temp)
#         # print(step)
#         temp[0] = temp[0] + step[0]
#         temp[1] = temp[1] + step[1]
        
#         if temp[0] < 0:
#             temp[0] = 0
#         if temp[0] >= self.dim_[0]:
#             temp[0] = self.dim_[0]-1
#         if temp[1] < 0:
#                 temp[1] = 0
#         if temp[1] >= self.dim_[1]:
#             temp[1] = self.dim_[1]-1
#         return temp

#     def get_action(self, _action):
#         if   _action[0] ==  0 and _action[1] == -1:
#             return 0 # S
#         elif _action[0] == -1 and _action[1] == -1:
#             return 1 # SW
#         elif _action[0] == -1 and _action[1] ==  0:
#             return 2 #  W
#         elif _action[0] == -1 and _action[1] ==  1:
#             return 3 # NW
#         elif _action[0] ==  0 and _action[1] ==  1:
#             return 4 # N
#         elif _action[0] ==  1 and _action[1] ==  1:
#             return 5 # NE
#         elif _action[0] ==  1 and _action[1] ==  0:
#             return 6 #  E
#         elif _action[0] ==  1 and _action[1] == -1:
#             return 7 # SE
#         else:
#             return 8 # Z
