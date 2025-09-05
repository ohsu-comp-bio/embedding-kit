def eucpair(v1, v2):
    '''
    calcuate the mean eucl distance for the pair of lists
    example: (list(1,3,4,5), list(6,7,8,9)) indicates euclidean distance between each pair of values (order matters)
    and then the mean euclidean distance of those computed
    '''   
    assert len(v1) == len(v2), 'two lists must be same length'
    euc_list = []
    for i in range(0, len(v1)):
        euc = np.linalg.norm(v1[i] - v2[i])
        euc_list.append(euc)
    mean_euc = statistics.mean(euc_list) 
    return round(mean_euc, 5)


def mhat(v1, v2):
    '''manhattan distance between two arrays'''
    return distance.cityblock(v1, v2)
