c. Classification System Development:
● Goal: Develop a system to classify the images based on something
non-trivial. Suggestion: You could try classifying whether the image is a
meme or not. Dataset for this is readily available as the positive class set
is the dataset given, and you can easily source non-memes from other
sources. You may freely choose any other classification task as well, but
keep in mind that sourcing labeled data for the same might not be as
easy. It is imperative that your classification task involves the provided
dataset in part or as a whole. Properly report your methodologies, findings
and performance of the model.

-----------------------------------------------------------------------
Methodology: This is a binary classification task with class labels being "meme"/"non-meme"
Environment: Google Colab(T4)

Data: For the dataset, randomly taking 5k datapoints from the hateful memes set. These would be for class "memes" (scripts_and_models/helper_movefiles.py)
For non-meme data,taking the MSCOCO validation set as the source of non-meme images.(http://images.cocodataset.org/zips/val2017.zip)

Using split-folders to set up the working directory in the ImageNet format with a 70/30 split.
Directory struct:
--test
----meme
----non-meme
--train
----meme
----non-meme

Main training script: main_training_script.ipynb
Approach: Finetuning a EfficientNet_B0 model. Will freeze the features branch and only set a new classifier head for 2 classes. Other architectures could be tried but I like working with effnet, its lightweight and accurate for most cases. 
Loss fn: CrossEntropyLoss
Optimiser: Adam with 0.001 learning rate
Was running out of gpu quota, or else could have tried with different hyper-params sweep
Input Transforms: Using efficient nets own transforms for the training

Model saving format: torchscript
Inference script: scripts_and_models/inference.py
