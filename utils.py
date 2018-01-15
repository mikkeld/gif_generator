from PIL import Image
import numpy as np
import urllib
import json
from scipy.misc import imresize
import bcolz


# sudo pip install http://effbot.org/media/downloads/Imaging-1.1.7.tar.gz
# http://pillow.readthedocs.io/en/3.4.x/reference/Image.html

def save_files(json_data, search_term, image_type='fixed_width_small'):
    """Downloads and saves gifs.

    Downloads gifs via giphy url. Saves downloaded gifs to folder gifs/

    Args:
        json_data (json): gify's api response.
        search_term (str): which term was used to search for gifs.
        image_type (str): gifys's response includes multiple gif formats.
            image_type can be one of these:
            fixed_height_still
            fixed_width_small
            fixed_width_small_still
            preview_webp
            fixed_height
            fixed_height_small_still
            480w_still
            downsized_medium
            preview
            preview_gif
            fixed_height_small
            fixed_width
            fixed_width_downsampled
            original_still
            fixed_height_downsampled
            downsized_small
            original_mp4
            downsized_still
            looping
            downsized_large
            fixed_width_still
            downsized
            original
    """
    for idx, gif in enumerate(json_data['data']):
        file_path = gif['images'][image_type]['url']
        if len(file_path) > 0:
            file = urllib.urlopen(file_path)
            path = 'gifs/{}_{}.gif'.format(search_term,idx)
            with open(path,'wb') as output:
                output.write(file.read())


def analyseImage(im):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    '''
    results = {
       'size': im.size,
       'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    im.seek(0)
    return results


def getFrames(im):
    '''
    Iterate the GIF, extracting each frame.
    '''
    mode = analyseImage(im)['mode'] 

    p = im.getpalette()
    last_frame = im.convert('RGBA')

    try:
        while True:
            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''
            if not im.getpalette():
                im.putpalette(p)

            new_frame = Image.new('RGBA', im.size)

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)

            new_frame.paste(im, (0,0), im.convert('RGBA'))
            yield new_frame

            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass


def processImage(path, reshape_to_vgg=False, image_limit=None):
    im = Image.open(path)
    frames = []
    for (i, frame) in enumerate(getFrames(im)):
        if image_limit and i == image_limit:
            break
        if reshape_to_vgg:
            frames.append(
                imresize(np.asarray(frame)[:,:,[0,1,2]],[224,224,3]))
        else:
            frames.append(np.asarray(frame))
        #frame.save('%s-%d.png' % (''.join(os.path.basename(path).split('.')[:-1]), i), 'PNG')
    return np.array(frames)


class Dataset(object):
    """Data management class."""
    def __init__(self, root_dir=None):
        if not root_dir:
            raise ValueError('Please specify a data location.')
        self.root_dir = root_dir 
        self.has_data = False
        self._remaining = None
        self._update()

    def add(self, data_arr):
        if self.has_data:
            self._append(data_arr)
        else:
            self._create(data_arr)
            
    def _create(self, data_arr):
        try:
            self.data = bcolz.carray(data_arr, rootdir=self.root_dir)
            self.data.flush()
            self.has_data=True
        except:
            raise
            
    def _update(self):
        try:
            self.data = bcolz.carray(rootdir=self.root_dir)
            self.has_data=True
        except IOError as err:
            self.has_data = False
            # print '[ERROR] There is no data in %s.'%(self.root_dir)
            
    def _append(self, data_arr):
        self._update()
        self.data.append(data_arr)
        self.data.flush()
    # TODO(tanozaslan) Change keyword replace, it is confusing with numpy build in
    # function.
    def reset_batch(self):
        self._remaining = None
        
    def get_random_batch(self, batch_size=10, replace=True):
        """Gives a random batch.
        Args:
            batch_size(int): number of examples in each batch.
            replace (boolean): If True, in each iteration gives a random batch.
                If False, iterated over the dataset, does not repeat an example
                until finishes one epoch. Last batch might be smaller than batch_size
                if batch_size is not perfect divider of number of examples in the
                dataset.
        """
        if batch_size > self.data.shape[0]:
                raise ValueError('Batch size cannot be bigger than data size.')
        data_size = self.data.shape[0]
        vector_len = self.data.shape[1]
        if replace:
            batch_idxs = np.random.choice(data_size, batch_size, replace=False)
        else:
            if not self._remaining:
                self._remaining = range(data_size)
            batch_idxs = []
            if len(self._remaining) < batch_size and len(self._remaining) > 0:
                batch_idxs = self._remaining
                self._remaining = []
            else:
                for _ in range(batch_size):
                    batch_idx = np.random.choice(range(len(self._remaining)),
                                                 size=[1],
                                                 replace=False)
                    batch_idxs.append(self._remaining.pop(batch_idx))
            if len(self._remaining) == 0:
                self.reset_batch()
        np.random.shuffle(batch_idxs)        
        data = self.data[batch_idxs,:,0:vector_len-1]
        target = self.data[batch_idxs,:,vector_len-1:vector_len]
        return data, target    