# Tutorial practice code based on "PyTorch Paper Replicating"
# https://www.learnpytorch.io/08_pytorch_paper_replicating/


# PyTorch modules
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary
# Helper functions(download dataset, random seed, result draw,,,)
from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
# ETC.
import matplotlib.pyplot as plt


def check_torch():
    # Check PyTorch installation
    assert int(torch.__version__.split(".")[1]) >= 12 or int(torch.__version__.split(".")[0]) == 2, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    return device


def make_dataloader(img_size, batch_size):
    # Download Datasets
    ## Download pizza, steak, sushi images from GitHub
    image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    print(f"Dataset Path {image_path}")

    # PyTorch Transforms
    # Transforms contains commmon transform works like resize·normalize·typecast,,,
    manual_transforms = transforms.Compose([transforms.Resize((img_size, img_size)) # Resizing
                                            , transforms.ToTensor()])               # Convert to Pytorch Tensor
    
    # DataLoader. Create data loaders
    # for data_setup, It use datasets.ImageFolder for creating DataLoader
    # datasets.ImageFolder useful when create Classification dataset with Folder-organized
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms, # use manually created transforms
        batch_size=batch_size
    )
    print(f"Labels: {class_names}")

    return train_dataloader, test_dataloader, class_names


def show_patch_embedding(image, image_size:int=224, batch_size:int=16):
    width = image_size
    height = image_size
    color_chan = 3
    patch_size = batch_size
    assert (width % patch_size == 0) and (height % patch_size == 0), "Image size must be divisible by patch size"

    # N: Number of Patch
    n_patch = int((width*height) / patch_size**2)

    # Embedding input shape
    embed_input_shape = (height, width, color_chan)
    # Embedding output shape
    embed_output_shape = (n_patch, (patch_size**2)*color_chan)

    # Patched information
    print(f"Input ({width}x{height}), Patch ({patch_size}x{patch_size}), Num of Patch : {n_patch}")
    print(f"Embedding input shape {embed_input_shape}")
    print(f"Embedding output shape {embed_output_shape}")

    # To handle with matplotlib, change shape
    # (chan, height, width) -> (height, width, chan)
    image_permuted = image.permute(1, 2, 0)

    # subplot for show
    fig, axs = plt.subplots(nrows=height // patch_size, # need int not float
                            ncols=width // patch_size,
                            figsize=(int(height / patch_size), int(width / patch_size)),
                            sharex=True,
                            sharey=True)

    # Show How to divide image into patch
    for i, patch_h in enumerate(range(0, height, patch_size)):
        for j, patch_w in enumerate(range(0, width, patch_size)):
            patch = image_permuted[patch_h:patch_h+patch_size   # Height(patch)
                                   , patch_w:patch_w+patch_size         # Width(patch)
                                   , :]                         # Chan(all)
            
            # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
            axs[i, j].imshow(patch) 

            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i+1,
                                rotation="horizontal",
                                horizontalalignment="right",
                                verticalalignment="center")
            axs[i, j].set_xlabel(j+1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()
    plt.show()


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dim:int=768):
        super().__init__()
        # 768 filters per 1 Patch
        # -> 768 filter per 768 length embedding vector 
        # 1 embedding vector has 768 features of patch
        self.conv2d = nn.Conv2d(in_channels=in_channels            # number of color channels
                                , out_channels=embedding_dim       # size D, (patch_size**2)*color_chan
                                , kernel_size=patch_size # could also use (patch_size, patch_size)
                                , stride=patch_size
                                , padding=0)
        # result shape example : torch.Size([1, 768, 14, 14])

        # convert previous embedding result
        # [batch, D, num patch(height), num patch(width) ]
        # to 
        # [batch, D, num patch]
        # Create flatten layer
        self.flatten = nn.Flatten(start_dim=2    # flatten num patch(height) (dimension 2)
                                  , end_dim=3)   # flatten num patch(width) (dimension 3)
        
        # for check input resolution correct
        self.patch_size = patch_size
        
    def forward(self, x):
        # Check size
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Do embedding process
        x_out_of_conv = self.conv2d(x)
        # -> [batch, D, num patch(height), num patch(width) ]
        embedding = self.flatten(x_out_of_conv)
        # -> [batch, D, num patch]

        embedding = embedding.permute(0, 2, 1) 
        # -> [batch, num patch, D] = [batch_size, N, P^2•C]

        return embedding


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, num_head:int=12, dropout_for_attention:float=0):
        super().__init__()

        # Layered Norm Layer
        self.layered_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Attention
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim
                                         , num_heads=num_head
                                         , dropout=dropout_for_attention
                                         , batch_first=True) # Input shape is [batch num, ...]
    
    def forward(self, x):
        x = self.layered_norm(x)
        out, _ = self.mha(query=x, key=x, value=x, need_weights=False)

        return out


class MultilayerPerceptronBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, mlp_size:int=3072, dropout:float=0.1):
        super().__init__()

        # Layered Norm Layer
        self.layered_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multilayer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim
                      , out_features=mlp_size)
            , nn.GELU()
            , nn.Dropout(p=dropout)
            , nn.Linear(in_features=mlp_size
                        , out_features=embedding_dim)
            , nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layered_norm(x)
        out = self.mlp(x)

        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim:int=768
                    , num_head:int=12, dropout_for_attention:float=0 # for MSA
                    , mlp_size:int=3072, dropout_for_mlp:float=0.1): # for MLP
        super().__init__()

        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim
                                                     , num_head=num_head
                                                     , dropout_for_attention=dropout_for_attention)
        
        self.mlp_block = MultilayerPerceptronBlock(embedding_dim=embedding_dim
                                                   , mlp_size=mlp_size
                                                   , dropout=dropout_for_mlp)

    def forward(self, x):
        x = self.msa_block(x) + x
        out = self.mlp_block(x) + x

        return out


class ViT(nn.Module):
    def __init__(self
                 # Input shape
                 , image_size:int=224
                 , in_channels:int=3
                 # for Embedding
                 , patch_size:int=16
                 , embedding_dim:int=768 
                 , dropout_for_embedding=0.1
                 # for MSA
                 , num_head:int=12
                 , dropout_for_attention:float=0
                 # for MLP
                 , mlp_size:int=3072
                 , dropout_for_mlp:float=0.1
                 # Number of Transformer Encoder
                 , num_transformer_layers:int=12
                 # Output shape
                 , num_classes:int=1000):
        super().__init__()

        # Check Image input size correct
        assert (image_size % patch_size == 0), "Image size must be divisible by patch size"

        # Number of Patches from image
        n_patch = int((image_size*image_size) / (patch_size**2))

        ################################## Embedding ##################################

        # Create an instance of patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels
                                              , patch_size=patch_size
                                              , embedding_dim=embedding_dim)
        
        # Class Token Embedding
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim)
                                            , requires_grad=True)
        
        # Positional Embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, n_patch+1, embedding_dim)
                                               , requires_grad=True)
        
        # Dropout for Embedding
        self.embedding_dropout = nn.Dropout(p=dropout_for_embedding)
        
        ################################## Transformer Encoder ##################################

        # Number of (num_transformer_layers) Transformer Encoders
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim
                                                                           , num_head=num_head
                                                                           , mlp_size=mlp_size
                                                                           , dropout_for_attention=dropout_for_attention
                                                                           , dropout_for_mlp=dropout_for_mlp) for _ in range(num_transformer_layers)])
        
        ################################## MLP Head ##################################

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim)
                                        , nn.Linear(in_features=embedding_dim, out_features=num_classes))
        
    def forward(self, x):
        # Expand Class Token Embedding
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1) # -1 means 'Do not expand this dim'

        ################################## Embedding ##################################
        # Patch Embedding
        x = self.patch_embedding(x)
        # Class Token Embedding
        x = torch.cat((class_token, x), dim=1)
        # Positional Embedding
        x = x + self.position_embedding
        # Dropout
        x = self.embedding_dropout(x)

        ################################## Transformer Encoder ##################################
        # Number of (num_transformer_layers) Transformer Encoders
        x = self.transformer_encoder(x)

        ################################## MLP Head ##################################
        # Classification input is z^0_L, 0 measn first rows of previouse output
        out = self.classifier(x[:, 0])

        return out


def train():
    # Check PyTorch status
    device = check_torch()

    # Train Param
    lr=3e-3
    adam_betas=(0.9, 0.999)
    weight_decay=0.3
    epochs = 10
    image_size = 224
    batch_size = 16

    # Load Dataset
    # Get Train/Test DataLoader
    train_dataloader, test_dataloader, class_names = make_dataloader(image_size, batch_size)

    # Model
    vit = ViT(num_classes=len(class_names))

    # Optimizer
    optimizer = torch.optim.Adam(params=vit.parameters()
                                 , lr=lr
                                 , betas=adam_betas
                                 , weight_decay=weight_decay)
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Random seed, default 42
    set_seeds()

    # Training
    results = engine.train(model=vit
                           , train_dataloader=train_dataloader
                           , test_dataloader=test_dataloader
                           , optimizer=optimizer
                           , loss_fn=loss_fn
                           , epochs=epochs
                           , device=device)
    
    # Plot Training result
    plot_loss_curves(results=results)
    plt.show()


if __name__ == "__main__":
    train()