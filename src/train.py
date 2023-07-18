import torch
import torch.nn as nn


NUM_CAPTIONS_PER_IMAGE = 5

# built-in function to use in training loop
def train_step(train_loader, extractor, 
               encoder, 
               decoder, 
               lr_rate,
               loss):
    """
        Concateneates block of model and performs the traing of model 
        Performs one epoch's training
    
    """

    # Configure the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    optimizer = torch.optim.Adam(decoder.parameters(),
                                 lr=lr_rate)
    # Initialize the batch loss
    batch_loss = 0.0

    # Train step on batchs
    for idx, (img_batch, caps_batch) in enumerate(train_loader):

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # MOVE THE USED (TENSOR,MODEL) TO DEFINED DEVICE
        #------

        # Make predictions for this batch
        img_embed = extractor(img_batch)
        encoder_output = encoder(img_embed)

        for i in range(NUM_CAPTIONS_PER_IMAGE):

            # Pass each of the five captions one by one to the decoder
            # along with the encoder outputs and compute the loss
            ith_caption = caps_batch[:, i, :]
            prediction = decoder(ith_caption, encoder_output)

            # Compute the loss
            loss = loss(prediction, ith_caption)
            batch_loss += loss

            # perform backpropagation and adjust learning weights
            loss.backward()
            optimizer.step()

            
    return {"loss": batch_loss/5}


class ImageCaptionModel(nn.Module):
    def __init__(self):
        super(ImageCaptionModel, self).__init__()
        pass