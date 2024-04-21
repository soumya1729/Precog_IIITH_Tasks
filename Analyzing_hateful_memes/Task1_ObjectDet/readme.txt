a. Object Detection:
● Goal: Utilize computer vision techniques to detect and identify objects
within the images of the memes.
● Tasks:
○ Apply object detection algorithms to identify various elements
within the meme images.
○ Catalog the types of objects detected and analyze their frequency
and distribution across the dataset.

--------------------------------------------------------
Problem with pre-trained models:
Direct Object Detection is not possible as most pretrained models like Yolo-x are built with MSCOCO datasets and have predefined 80/90/100 class-list. 
Newer architectures like GroundingDINO allow for zero-shot det on both COCO and novel human input labels. 
But still, that would require actual human labels but within the hateful dataset, we dont know which objects to specifically look for unless we go manually go through the 12k(12140) images.

Problem with building model from scratch:
This was definitely doable, at least with a subset of the 12k(12140) dataset by manually annotating it. However, I chose to go with a different approach.

Methodology:
************************************************************************
Without ground truth labels, I am modelling this as an unsupervised-task.
************************************************************************

I wanted to treat object detection as a VQA task with IDEFICS2(built on LLAMA V1 and OpenCLIP), but despite my best efforts could not get it to run either on Kaggle(T4 X2) or Colab(Basic/free tier) or even on Lightning AI Studio(Free tier only has A100 single node which was not enough). I tried with lower precision like 4bit and 8bit but the outputs were inconsistent.

Hence, I discarded IDEFICS for the moment(Definitely a future scope). My current pipeline is:
1) First phase-Image captioning through prompted conditional generation(basically vqa).Generating captions for all 12k(12140) images in the dataset.Expected output-> a more or less decent description of the scene for each image. Model used: Salesforce BLIP model(blip2-opt-2.7b) full precision(4bit and 8bit models were inconsistent, have raised this issue on their huggingface modelcard)
>> A crocodile, a woman, and a desk (Row no: 6076 in the below file)
Outputs of this phase stored in file: "output/image_caption.csv"
2) Keyword identification-Chose to keep it simple and extract only the nouns. Expect output-> a list of tokens(lemmatised form taken) which ideally act as objects present in the scene.Library used: Spacy.
>> ['crocodile', 'woman', 'desk'] (Row no: 6076 in the below file)
Outputs of this phase stored in file: "output/image_caption_labels.csv"
Future scope: Since the precog task was originally on object det, I will try to include object det in here as a filter to phase 1. Will run a GroundingDINO/Owl-ViT to run object det on the images with the tokens generated acting as labels. Only those labels crossing maybe 80% threshold would be treated as actual objects in the scene. Other objects failing to cross the threshold will be deemed faulty text generated from phase1
3) All the objects in all the images are pooled together for a freq dist.
Outputs of this phase stored in file: "output/frequency_distribution.csv"

Approximately 48k tokens were pooled together into 4020 unique classes->Indicating 4020 objects with multiple instances. The freq dist shows their break-up.

Environment Used: Kaggle Notebook with Accelerator: GPU T4 X2


Limitation:
A) In some cases, the image caption generation is muddled up with the text overlaid on the image. For example in case of Row:5945 in the image_caption.csv, filepath: /kaggle/input/facebook-hateful-meme-dataset/data/img/10895.png, the caption generated is : A mexican without a car(This was actually part of the text, while the image was of a man with a mexican hat).
In phase2, this got labeled into mexican and car.
Had I run objet det now with threshold, I would have gotten mexican and car should have been filtered out. 
So, this is not that bad an approach. However, will work better without text in Task2

B) If in a image of a dog, the caption generated for some reason is: a dog, a dog, a dog. Then according to my pipeline right now, it may seem like there are 3 dogs as the final class list would be ['dog','dog','dog']. I cannot take a set of this blindly as there may indeed be 2 dogs in this picture. However with object det included after step2, I can check for the number of bounding boxes and determine how many dogs are there actually. 
=======================================================================

Future Scope: Currently, since I have not manually annotated the whole 12k(12140) dataset, I do not have ground-truth labels to compare the output of my pipeline with. This would take a manual verification, which if time permits I will do. But for now, without ground truth labels, there is no really good metric to gauge accuracy of this setup.
