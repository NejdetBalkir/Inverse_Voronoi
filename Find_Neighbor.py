# This function find the neighbors of each cell

def find_neighbor(vor):


    neighbor_storage = []
    neighbor_vertices_storage = []

    for region in vor.point_region:
        neighborhood = []
        
        vertices_of_region = vor.regions[region]
        neighborhood.append(region)

        for otherRegion in vor.point_region:
        # for otherRegion in point_region:
            if (region != otherRegion):
                vertices_of_otherRegion = vor.regions[otherRegion]
                common_elements = list(set(vertices_of_region) & set(vertices_of_otherRegion))
                num_common_elements = len(common_elements)
                
                if (num_common_elements >= 2):
                    if (num_common_elements >= 3):
                        for s in range(num_common_elements):
                            if (common_elements[s] == -1):
                                negative_value = s
                        common_elements.remove(common_elements[negative_value])
                    check_list = [otherRegion,region,common_elements]
                    if check_list not in neighbor_vertices_storage:
                        neighborhood.append(otherRegion)
                        neighbor_vertices = []
                        neighbor_vertices.append(region)
                        
                        neighbor_vertices.append(otherRegion)
                        neighbor_vertices.append(common_elements)
                        # print("neighbor vertices:" , neighbor_vertices)
                    
                        neighbor_vertices_storage.append(neighbor_vertices)
        neighbor_storage.append(neighborhood)



    # print("neighborhood storage:" , neighbor_storage)
    # print("neighbor vertices storage:" , neighbor_vertices_storage)

    return neighbor_storage, neighbor_vertices_storage