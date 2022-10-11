import os.path
from useg.utils import isimage
import numpy as np
import skimage.io
import skimage.segmentation

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

def presegment(input_files, segmentation_directory, segmentation_args, force = False):
    preseg_files = []
    if(os.path.exists(segmentation_directory)==False or force==True):
        if(os.path.exists(segmentation_directory)==False):
            os.mkdir(segmentation_directory)

        for i in range(len(input_files)):
            
            image = skimage.io.imread(input_files[i])
            image = image[:, :, [3,2,1]]
            for c in range(image.shape[-1]):
                image[:, :, c] = (image[:, :, c]-image[:, :, c].min())/(image[:, :, c].max()-image[:, :, c].min())
            preseg = skimage.segmentation.slic(image, **segmentation_args)

            preseg_path = os.path.join(segmentation_directory, 'image_'+str(i)+'.tif')
            skimage.io.imsave(preseg_path, preseg)
            assert np.prod(preseg==skimage.io.imread(preseg_path))==1, 'I/O error'



            preseg_files.append(preseg_path)
    else:
        print('Segmentation results already exists')
        for i in range(len(input_files)):
            preseg_path = os.path.join(segmentation_directory, 'image_'+str(i)+'.tif')
            preseg_files.append(preseg_path)
    
    return(preseg_files)





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