import torch
from blocks import TransformerEncoderBlock, TransformerDecoderBlock, FeatureExtractor
from prepare_dataset import Flickr8KDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from train import train_step


def main():
    # Path to the images
    IMAGES_PATH = "/home/okan/Desktop/Image_Caption/data/Flicker8k_Dataset"
    CAPTION_FILE = '/home/okan/Desktop/Image_Caption/data/Flickr8k.token.txt'

    # Desired image dimensions
    IMAGE_SIZE = (299, 299)

    # Vocabulary size
    VOCAB_SIZE = 10000

    # Fixed length allowed for any sequence
    SEQ_LENGTH = 25

    # Dimension for the image embeddings and token embeddings
    EMBED_DIM = 512

    # Per-layer units in the feed-forward network
    FF_DIM = 512

    # Other training parameters
    BATCH_SIZE = 64
    EPOCHS = 10
    NUM_CAPTIONS_PER_IMAGE = 5

    TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_data = Flickr8KDataset(IMAGES_PATH, CAPTION_FILE, transform=TRANSFORM)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        print(("Epoch:", epoch))

        train_step(train_loader=train_dataloader, 
                   extractor=FeatureExtractor(),
                   encoder=TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1),
                   decoder=TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2),
                   lr_rate=0.01,
                   loss=loss_fn)

    

if __name__ == "__main__":
    test_data = main()
    # test_data[25]
    

    # torch.manual_seed(42)
    # test_input = torch.rand((3,255,255))
    # test_input = test_input.unsqueeze(0)
    # ext = FeatureExtractor()
    # last_output = ext(test_input)

    # last_output = last_output.view((2, -1))
    # last_output.shape
    # last_output.view()

