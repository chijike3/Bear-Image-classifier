
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
pip install bing-image-downloader
from bing_image_downloader import downloader
from fastai.vision.all import *

# Download images of "cats"
downloader.download(query="teddy_bear", limit=150, output_dir='teddy_bear_image', adult_filter_off=True)
downloader.download(query="grizlly_bear", limit=150, output_dir='grizlly_bear_image', adult_filter_off=True)
downloader.download(query="black_bear", limit=150, output_dir='black_bear_image', adult_filter_off=True)


# Get the list of image files from each directory
teddy_bear_ims = get_image_files('teddy_bear_image')
grizlly_bear_ims = get_image_files('grizlly_bear_image')
black_bear_ims = get_image_files('black_bear_image')

# Combine the lists
all_bear_images = teddy_bear_ims + grizlly_bear_ims + black_bear_ims

# You can now work with the 'all_bear_images' list
for image in all_bear_images:
      one_image = Image.open(image)
      show_image(one_image)     #will print all the image in the, all_ims list

# create a data Datablock object 
bear_datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)

dls = bear_datablock.dataloaders('/content/')
dls.valid.show_batch(max_n=4, nrows=1)

# The RandomResizedCrop method shows different size of an image to the model to enable it learn the basic concept of what an image is. This is also called Data augumentation
bear_datablock = bear_datablock.new(item_tfms=RandomResizedCrop(128),  batch_tfms=aug_transforms(mult=2))
dls = bear_datablock.dataloaders('/content/')
dls.train.show_batch(max_n=8, nrows=1, unique=True)

# Now train and fine tune the model as follows
bear_datablock_learn = vision_learner(dls, resnet18, metrics=error_rate)
bear_datablock_learn.fine_tune(4)

# plot a confusion matrix to check the performance of the model after training
bear_datablock_interp = ClassificationInterpretation.from_learner(bear_datablock_learn)
bear_datablock_interp.plot_confusion_matrix()
# visualise the top 5 losses
bear_datablock_interp.plot_top_losses(5, nrows=2)

