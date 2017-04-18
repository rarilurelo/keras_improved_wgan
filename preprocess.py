import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img
from PIL import Image

class DataFeeder(object):
    def __init__(self, load_dir, batch_size=64, size=(64, 64)):
        self.load_dir = load_dir
        self.batch_size = batch_size
        self.size = size
        self.generator = ImageDataGenerator(data_format='channels_first').flow_from_directory(self.load_dir, target_size=size, batch_size = batch_size)

    def fetch_data(self):
        print('hoge')
        data, _ = next(self.generator)
        if data.shape[0] == self.batch_size:
            return data/255.
        else:
            return self.fetch_data()

    def save_images(self, arrays, names, concat, save_dir='save'):
        if not isinstance(names, list):
            names = [names]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not concat:
            for array, name in zip(arrays, names):
                image = array_to_img(array, data_format='channels_first').resize((60, 60))
                image.save("{}.png".format(name), quality=100)
        else:
            canvas = Image.new('RGB', (60*len(arrays), 60), (255, 255, 255))
            for i, array in enumerate(arrays):
                image = array_to_img(array, data_format='channels_first').resize((60, 60))
                canvas.paste(image, (i*60, 0))
            canvas.save(os.path.join(save_dir, "{}.png".format(names[0])), quality=100)

