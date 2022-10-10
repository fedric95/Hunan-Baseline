import os
from useg.utils import isimage
import numpy as np


def read_dataset(input_dir, sensor, split):
    
    assert split in ['train', 'val', 'test'], 'Invalid split'
    assert sensor in ['s1', 's2'], 'Invalid sensor'
    

    input_dir =  os.path.join(input_dir, split)
    input_files = os.listdir(os.path.join(input_dir, sensor))
    input_files = [file for file in input_files if isimage(os.path.join(input_dir, sensor, file))]
    
    label_files = []
    for input_file in input_files:
        label_file = os.path.join(input_dir, 'lc', input_file.replace(sensor, 'lc'))
        input_file = os.path.join(input_dir, sensor, input_file)
        if(os.path.exists(label_file)):
            label_files.append(label_file)
        else:
            raise Exception('File not found')

    input_files = [os.path.join(input_dir, sensor, file) for file in input_files]
    return(input_files, label_files)


class HunanTransform:
    def __init__(self):
        self.igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255])
    def __call__(self, out):
        out['label'][out['label'] == 255] = 12
        out['label'] = self.igbp2hunan[out['label']]
        out['image'] = out['image'][[3,2,1], :, :] #True Color
        
        out['label'][out['label']==7]=255
        out['label'][out['label']==255] = 0

        out['label'][out['label']==2] = 6 # grassland -> others
        out['label'][out['label']==3] = 6 # wetland -> others
        out['label'][out['label']==5] = 6 # bare lands -> others

        out['label'][out['label']==4] = 2
        out['label'][out['label']==6] = 3

        out['mask'] = np.isin(out['label'], [0,1,2,3,4,5,6])

        return(out)



class HunanTransform:
    def __init__(self):
        self.igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255])
    def __call__(self, out):
        out['label'][out['label'] == 255] = 12
        out['label'] = self.igbp2hunan[out['label']]
        out['image'] = out['image'][[3,2,1], :, :] #True Color
        
        out['label'][out['label']==7]=255

        out['mask'] = out['label']!=255
        
        out['label'][out['label']==2] = 6 # grassland -> others
        out['label'][out['label']==3] = 6 # wetland -> others
        out['label'][out['label']==5] = 6 # bare lands -> others
        out['label'][out['label']==4] = 2
        out['label'][out['label']==6] = 3
        out['label'][out['label']==255] = 0
        


        for c in range(out['image'].shape[0]):
            out['image'][c, :, :] = (out['image'][c, :, :]-out['image'][c, :, :].min())/(out['image'][c, :, :].max()-out['image'][c, :, :].min())

        #import matplotlib.pyplot as plt
        #plt.imshow(np.transpose(out['image'], (1, 2, 0)))
        #plt.show()

        return(out)