b. Caption Impact Assessment:
● Goal: Assess the effect of overlaid captions on the accuracy and
effectiveness of object detection.
● Tasks:
○ Determine how text overlays influence the object detection
process.
○ If necessary, develop and implement methods to minimize the
impact of captions, such as using image processing techniques to
filter out text. (You are not expected to make the model for this, try
to find models that can do this for you)


--------------------------------------------------------
This would be a continuation of Task2. However, since I have not manually annotated the whole 10k dataset, I do not have ground-truth labels to compare the accuracy from task2 with task1. With the way I have approached this, there is only one way to compare Task1 and Task2 approaches i.e through the number of labels generated and a cursory manual verification on random samples.

For this Task: I have added one preprocessing step. I wanted to use SAM-OCR(https://github.com/yeungchenwa/OCR-SAM) for inpainting with SAM and Stable-Diffusion based approach. But found the approach a bit of overkill as it consisted of downloading too many models. I decided against it considering resource and time constraints. I had to remove the overlaying text from 10k images and my colab and kaggle gpu quotas were limited.

Methodolgy:
I decided to use Keras-OCR to detect and recognize text in the images, created masks of the regions and inpainted with opencv Navier-Stokes. This way all the 10k images of the hateful dataset are filtered into 10k clean images. However, keras ocr uses the CRAFT text det module and it is pretty slow. With lowering GPU quota on Kaggle, I had to scale down the images into 300x300 for faster processing.

Libraries Requirements:
transformers==4.38.2
keras-ocr==0.9.3
keras==2.15.0
tensorflow==2.15.0

Scripts: Script present in "scripts" dir.
Data: The filtered dataset was uploaded to Kaggle datasets and made public:FacebookHatefulDataset_WithoutTextCaptions(Uploaded to drive too: https://drive.google.com/file/d/1C7uZ0LuwrSEuLKP2j7sKc9Nrw2eJTawB/view?usp=sharing)

Steps: 
1)Filtering out text from images(kerasOcr_FilterOutText.ipynb) 
2) The remaining steps are similar to Task1 where I use this filtered dataset as input to the 3-phase pipeline.(task2precog-blip2-spacy.ipynb)

Output: 51263 tokens are pooled in the final stage->2234 unique classes with multiple instances of each class.Therefore, it can be considered there are 2234 unique objects acc to our pipeline in the whole dataset. All the output files including the frequency dict is stored in the `data` dir.

-----------------------------------------------------------
The text overlays then has a definite impact as is evident from the final list of objects.
No of Objects via Task1: 4020
No of Objects via Task2: 2234 

Without actual ground truth labels, difficult to say with the current approach about the accuracy of either approaches.
