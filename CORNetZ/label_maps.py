
finelabel_to_coarselabel_temp = {
    ('beaver', 'dolphin', 'otter', 'seal', 'whale')                     : 'aquatic mammals',
    ('aquarium_fish', 'flatfish', 'ray', 'shark', 'trout')              : 'fish',
    ('orchid', 'poppy', 'rose', 'sunflower', 'tulip')                   : 'flowers',
    ('bottle', 'bowl', 'can', 'cup', 'plate')                           : 'food containers',
    ('apple', 'mushroom', 'orange', 'pear', 'sweet_pepper')             : 'fruits and vegetables',
    ('clock', 'keyboard', 'lamp', 'telephone', 'television')            : 'household electrical devices',
    ('bed', 'chair', 'couch', 'table', 'wardrobe')                      : 'household furniture',
    ('bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach')          : 'insects',
    ('bear', 'leopard', 'lion', 'tiger', 'wolf')                        : 'large carnivores',
    ('bridge', 'castle', 'house', 'road', 'skyscraper')                 : 'large man-made outdoor things',
    ('cloud', 'forest', 'mountain', 'plain', 'sea')                     : 'large natural outdoor scenes',
    ('camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo')           : 'large omnivores and herbivores',
    ('fox', 'porcupine', 'possum', 'raccoon', 'skunk')                  : 'medium-sized mammals',
    ('crab', 'lobster', 'snail', 'spider', 'worm')                      : 'non-insect invertebrates',
    ('baby', 'boy', 'girl', 'man', 'woman')                             : 'people',
    ('crocodile', 'dinosaur', 'lizard', 'snake', 'turtle')              : 'reptiles',
    ('hamster', 'mouse', 'rabbit', 'shrew', 'squirrel')                 : 'small mammals',
    ('maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree') : 'trees',
    ('bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train')           : 'vehicles 1',
    ('lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor')            : 'vehicles 2',
}
finelabel_to_coarselabel = {}
for k, v in finelabel_to_coarselabel_temp.items():
    for key in k:
        finelabel_to_coarselabel[key] = v


finelabel_dict = {
    'apple'              : 0, 
    'aquarium_fish'      : 1,
    'baby'               : 2,
    'bear'               : 3,
    'beaver'             : 4,
    'bed'                : 5,
    'bee'                : 6,
    'beetle'             : 7, 
    'bicycle'            : 8,
    'bottle'             : 9 ,
    'bowl'               : 10 ,
    'boy'                : 11,
    'bridge'             : 12 ,
    'bus'                : 13 ,   
    'butterfly'          : 14,   
    'camel'              : 15,   
    'can'                : 16,   
    'castle'             : 17, 
    'caterpillar'        : 18 ,   
    'cattle'             : 19 , 
    'chair'              : 20 ,   
    'chimpanzee'         : 21 , 
    'clock'              : 22 ,
    'cloud'              : 23 , 
    'cockroach'          : 24 ,   
    'couch'              : 25 ,
    'crab'               : 26 ,   
    'crocodile'          : 27 ,   
    'cup'                : 28 ,   
    'dinosaur'           : 29 ,   
    'dolphin'            : 30 ,
    'elephant'           : 31 ,
    'flatfish'           : 32 ,   
    'forest'             : 33 ,
    'fox'                : 34 ,
    'girl'               : 35 ,
    'hamster'            : 36 ,
    'house'              : 37 ,
    'kangaroo'           : 38 ,
    'keyboard'           : 39 ,
    'lamp'               : 40 ,
    'lawn_mower'         : 41 ,
    'leopard'            : 42 ,
    'lion'               : 43 ,
    'lizard'             : 44 ,
    'lobster'            : 45 ,
    'man'                : 46 ,
    'maple_tree'         : 47 ,
    'motorcycle'         : 48 ,
    'mountain'           : 49 ,
    'mouse'              : 50 ,
    'mushroom'           : 51 ,
    'oak_tree'           : 52 ,
    'orange'             : 53 ,
    'orchid'             : 54 ,
    'otter'              : 55 ,
    'palm_tree'          : 56 ,
    'pear'               : 57 ,
    'pickup_truck'       : 58 ,
    'pine_tree'          : 59 ,
    'plain'              : 60 ,
    'plate'              : 61 ,
    'poppy'              : 62 ,
    'porcupine'          : 63 ,
    'possum'             : 64 ,
    'rabbit'             : 65 ,
    'raccoon'            : 66 ,
    'ray'                : 67 ,
    'road'               : 68 ,
    'rocket'             : 69 ,
    'rose'               : 70 ,
    'sea'                : 71 ,
    'seal'               : 72 ,
    'shark'              : 73 ,
    'shrew'              : 74 ,
    'skunk'              : 75 ,
    'skyscraper'         : 76 ,
    'snail'              : 77 ,
    'snake'              : 78 ,
    'spider'             : 79 ,
    'squirrel'           : 80 ,
    'streetcar'          : 81 ,
    'sunflower'          : 82 ,
    'sweet_pepper'       : 83 ,
    'table'              : 84 ,
    'tank'               : 85 ,
    'telephone'          : 86 ,
    'television'         : 87 ,
    'tiger'              : 88 ,
    'tractor'            : 89 ,
    'train'              : 90 ,
    'trout'              : 91 ,
    'tulip'              : 92 ,
    'turtle'             : 93 ,
    'wardrobe'           : 94 ,
    'whale'              : 95 ,
    'willow_tree'        : 96 ,
    'wolf'               : 97 ,
    'woman'              : 98 ,
    'worm'               : 99 ,
}


coarse_labels = ['aquatic mammals', 'fish', 'flowers', 'food containers',
                 'fruits and vegetables', 'household electrical devices',
                 'household furniture', 'insects', 'large carnivores',
                 'large man-made outdoor things', 'large natural outdoor scenes',
                 'large omnivores and herbivores', 'medium-sized mammals',
                 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
                 'trees', 'vehicles 1', 'vehicles 2'
]

coarselabel_dict = {
    'aquatic mammals'               :0, 
    'fish'                          :1, 
    'flowers'                       :2, 
    'food containers'               :3, 
    'fruits and vegetables'         :4, 
    'household electrical devices'  :5, 
    'household furniture'           :6, 
    'insects'                       :7, 
    'large carnivores'              :8, 
    'large man-made outdoor things' :9, 
    'large natural outdoor scenes'  :10, 
    'large omnivores and herbivores':11, 
    'medium-sized mammals'          :12, 
    'non-insect invertebrates'      :13, 
    'people'                        :14, 
    'reptiles'                      :15, 
    'small mammals'                 :16, 
    'trees'                         :17, 
    'vehicles 1'                    :18, 
    'vehicles 2'                    :19, 
}


fine_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
]

def get_fine_labels():
    return fine_labels

def get_coarse_labels():
    return coarse_labels

def finelabel_from_idx(idx):
    return fine_labels[idx]

def coarselabel_from_idx(idx):
    return coarse_labels[idx]

def coarselabels_from_fineidxs(idxs):
    c = list()
    for idx in idxs:
        c.append(coarselabel_from_fineidx(idx))
    return c 

def idx_from_finelabel(fine_label):
    return finelabel_dict[fine_label]

def idx_from_coarselabel(coarse_label):
    return coarselabel_dict[coarse_label]

def idxs_from_coarselabels(coarse_labels):
    i = list()
    for label in coarse_labels:
        i.append(idx_from_coarselabel(label))
    return i 

def coarselabel_from_finelabel(fine_label):
    return finelabel_to_coarselabel[fine_label]

def coarselabel_from_fineidx(fine_idx):
    fine_label = finelabel_from_idx(fine_idx)
    return coarselabel_from_finelabel(fine_label)



