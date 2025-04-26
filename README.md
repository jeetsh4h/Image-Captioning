# Objective

The objective of this project is to build a model that can generate captions for images.

The directory structure of this project is shown below:
```bash
root/
├── coco/
│   ├── annotations/
│   │   ├── captions_train2014.json
│   │   └── captions_val2014.json
│   ├── karpathy/
│   │   └── dataset_coco.json
│   └── test2014/
│   ├── train2014/
│   └── val2014/
│
├── images/
├── pretrained/
├── results/
│
├── caption.py
├── datasets.py
├── evaluation.py
├── models.py
├── README.md
├── train.py
└── utils.py
```
# Model

I use Encoder as Efficientnet to extract features from image and Decoder as Transformer to generate caption. But I also change the attention mechanism at step attention encoder output. Instead of using the Multi-Head Attention mechanism, I use the Attention mechanism each step to attend image features.
<figure align="center">
  <p align="center"><img src="./images/model_architecture_trans.png" width="600"/>
    <figcaption><b>Model architecture:</b> <i>The architecture of the model Image Captioning with Encoder as Efficientnet and Decoder as Transformer</i></figcaption>
  </p>
</figure>

# Dataset
We are using the MSCOCO '14 Dataset. You'd need to download the Training (13GB),  Validation (6GB) and Test (6GB) splits from [MSCOCO](http://cocodataset.org/#download) and place them in the `./coco` directory.

We are also using Andrej Karpathy's split of the MSCOCO '14 dataset. It contains caption annotations for the MSCOCO, Flickr30k, and Flickr8k datasets. You can download it from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). You'd need to unzip it and place it in the `./coco/karpathy` directory.
In Andrej's split, the images are divided into train, val and test sets with the number of images in each set as shown in the table below:

| Image/Caption | train | val | test |
| :--- | :--- | :--- | :--- |
| Image | 113287 | 5000 | 5000 |
| Caption | 566747 | 25010 | 25010 |



# Training and Validation
## Pre-processing
### Images
I preprocessed the images with the following steps:
- Resize the images to 256x256 pixels.
- Convert the images to RGB.
- Normalize the images with mean and standard deviation.
I normalized the image by the mean and standard deviation of the ImageNet images' RGB channels.
```python
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```


### Captions
Captions are both the target and the inputs of the Decoder as each word is used to generate the next word.

I use BERTTokenizer to tokenize the captions.
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token = tokenizer(caption, max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"][0]
```

For more details, see `datasets.py` file.

## Training
### Model configs

- embedding_dim: 512
- vocab_size: 30522
- max_seq_len: 128
- encoder_layers: 6
- decoder_layers: 12
- num_heads: 8
- dropout: 0.1

### Hyperparameters

- n_epochs: 25
- batch_size: 24
- learning_rate: 1e-4
- optimizer: Adam
- adam parameters: betas=(0.9, 0.999), eps=1e-9
- loss: CrossEntropyLoss
- metric: bleu-4
- early_stopping: 5

## Validation
I evaluate the model on the validation set after each epoch. For each image, I generate a caption and evaluate the BLEU-4 score with list of reference captions by sentence_bleu. And for all the images, I calculate the BLEU-4 score with the corpus_bleu function from NLTK.

You can see the detaile in the `train.py` file. Run `train.py` to train the model.
```bash
python train.py
```

> All defaults mentioned above are used. Use `--help` for options.

# Evaluation
See the `evaluation.py` file. Run `evaluation.py` to evaluate the model.

```bash
python evaluation.py
```
> All defaults mentioned above are used. Use `--help` for options.

To evaluate the model, I used the [pycocoevalcap package](https://github.com/salaniz/pycocoevalcap). Install it by `pip install pycocoevalcap`. And this package need to be Java 1.8.0 installed.

```bash
sudo apt-get update
sudo apt-get install openjdk-8-jdk
java -version

pip install pycocoevalcap
```
> Ensure that the output of `java -version` is `1.8.*`

Moreover, due to the package not being maintained, one has to manually comment out `SPICE` score calulation by editing the package file.

We used beam search to generate captions with beam size of 3 and 5. The metrics used to evaluate the model are BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, and CIDEr. The results on the test set (5000 images) are shown below.

Metrics extracted from `our_model_3/results/metrics_summary.txt`:

| Beam Width | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | METEOR | ROUGE_L | CIDEr |
|:----------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-----:|
| 3          | 0.3134 | 0.1815 | 0.1056 | 0.0636 | 0.1354 | 0.3116  | 0.1234 |
| 5          | 0.2935 | 0.1737 | 0.1040 | 0.0639 | 0.1538 | 0.2974  | 0.1055 |

# Inference
See the file `caption.py`. Run `caption.py` to generate captions for the test images. If you don't have resources for training, you can email me at [jeet.c.shah@flame.edu.in](mailto:jeet.c.shah@flame.edu.in) for pre-trained weights.

```bash
python caption.py -i <image_dir/image-file>
```
> For interactive mode, drop the `-i` argument
> All defaults mentioned above are used. Use `--help` for options.

```python
from evaluation import generate_caption

cap = generate_caption(
    model=model,
    image_path=image_path,
    transform=transform,
    tokenizer=tokenizer,
    max_seq_len=args.max_seq_len,
    beam_size=args.beam_size,
    device=device
)
print("--- Caption: {}".format(cap))
```

**Some examples of captions generated from COCO images are shown below.**
<table>
  <tr>
    <td><img src="images/test_1.jpg" ></td>
    <td><img src="images/test_2.jpg" ></td>
    <td><img src="images/test_5.jpg" ></td>
  </tr>
  <tr>
    <td>a bride and groom cutting into a cake with a knife</td>
    <td>a man riding a wave on top of a surfboard</td>
    <td>a keyboard mouse and mouse are on a desk</td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="images/test_3.jpg" ></td>
    <td><img src="images/test_4.jpg" ></td>
    <td><img src="images/test_7.jpg" ></td>
  </tr>
  <tr>
    <td>a red fire hydrant sitting on the side of a road</td>
    <td>a woman holding a hot dog in her hands</td>
    <td>there is a stop sign on the side of the road</td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="images/test_6.jpg" ></td>
    <td><img src="images/test_11.jpg" ></td>
    <td><img src="images/test_9.jpg" ></td>
  </tr>
  <tr>
    <td>a little girl sitting at a table with a laptop</td>
    <td>two girls are playing frisbee in a field</td>
    <td>a group of people on a beach with surfboards</td>
  </tr>
</table>

**Some examples of captions generated from other images that are not in the COCO dataset are shown below.**
  <table>
  <tr>
    <td><img src="images/pogba.jpg" ></td>
    <td><img src="images/baby.jpg" ></td>
    <td><img src="images/test.jpg" ></td>
  </tr>
  <tr>
    <td>a man on a soccer field kicking a ball</td>
     <td>there is a baby that is laying on a bed</td>
     <td>there is a dog that is standing in the grass</td>
  </tr>
 </table>
