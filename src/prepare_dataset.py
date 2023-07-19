import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

from PIL import Image
import os
import numpy as np
import re

SEQ_LENGTH = 25

class Flickr8KDataset(Dataset):

    def __init__(self, root_dir, caption_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load the captions file and create a mapping
        self.captions = self.load_captions(caption_file)


        # Get the list of image filenames
        self.image_filenames = list(self.captions[0].keys())
        
        
        
        # Pre-tranied sentencepiece model along with corresponding vocabulary
        self.vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
        self.spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

    def load_captions(self, caption_file):
        # Implement your code to load the captions from the file
        # and create a mapping of image filenames to their captions

        with open(caption_file) as caption_file:
            caption_data = caption_file.readlines()
            caption_mapping = {}
            text_data = []
            images_to_skip = set()

            for line in caption_data:
                line = line.rstrip("\n")
                # Image name and captions are separated using a tab
                image_name, caption = line.split("\t")

                # Each image is repeated five times for the five different captions.
                # Each image name has a suffix `#(caption_number)`
                image_name = image_name.split('#')[0]
                image_name = os.path.join("Flicker8k_Dataset", image_name.strip())

                # We will remove caption that are either too short to too long
                tokens = caption.strip().split()
                
                if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                    images_to_skip.add(image_name)
                    continue

                if image_name.endswith("jpg") and image_name not in images_to_skip:
                    # We will add a start and an end token to each caption
                    caption = "<bos> " + caption.strip() + " <eos>"
                    text_data.append(caption)

                    if image_name in caption_mapping:
                        caption_mapping[image_name].append(caption)
                    else:
                        caption_mapping[image_name] = [caption]

            for img_name in images_to_skip:
                if img_name in caption_mapping:
                    del caption_mapping[img_name]

        return caption_mapping, text_data
    
    def custom_standardization(self, input_string):
        strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        strip_chars = strip_chars.replace(">", "")

        
        lowercase = input_string.lower()
        
        return re.sub("[%s]" % re.escape(strip_chars), "", lowercase)
    
    def split_dataset(self,rate=0.2, shuffle=True,
                      seed=42):
        dataset_size = len(self.image_filenames)
        indices = list(range(dataset_size))
        split = int(np.floor(rate * dataset_size))
        if shuffle:
            np.random.seed(seed=seed)
            np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        
        return train_indices, val_indices

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):


        filename = str(self.image_filenames[index])

        

        # Load the image
        image = Image.open(os.path.join('/home/okan/Desktop/Image_Caption/data/', filename)).convert("RGB")

        # Apply transformation for image if specified
        if self.transform is not None:
            image = self.transform(image)

        # Get the corresponding caption then standardize and apply text transformation
        caption = self.captions[0][filename]
        


        for idx, str_text in enumerate(caption):
            standardizated_caption = self.custom_standardization(str_text)
            

            text_transform = T.Sequential(
                T.SentencePieceTokenizer(self.spm_model_path),
                T.VocabTransform(load_state_dict_from_url(self.vocab_path)),
                T.Truncate(max_seq_len=SEQ_LENGTH),
                T.ToTensor(),
                T.PadTransform(max_length=SEQ_LENGTH, pad_value=0)
                )
            caption[idx] = text_transform(standardizated_caption)


        
        
        # Return the image and caption as tensors
        return image, caption
    


# ROOT_DIR = '/home/okan/Desktop/Image_Caption/data/Flicker8k_Dataset'
# CAPTION_FILE = '/home/okan/Desktop/Image_Caption/data/Flickr8k.token.txt'


# TRANSFORM = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# dataset = Flickr8KDataset(ROOT_DIR, CAPTION_FILE, TRANSFORM)



# dataset[:2] # birden çok yine göstermiyor bu hataya sebep olabilir mi?

# # HATA VERİNİN HAZIRLANMASI AŞAMASINDA ,HALA CAPTIONLAR TENSOR ŞEKLİNDE ÇIKMIYOR, LİSTE ŞEKLİNDE ÇIKIYOR,(HER BİR CAPTION TENSOR OLSA DA LİSTE NİN İÇİNDELER)

