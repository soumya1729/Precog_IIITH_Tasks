import shutil, random, os
dirpath = '/hateful_memes/img'#set this to point to the hateful_memes/img folder
destDirectory = '/binary_classification/meme' #set this to the output dir to store 5k meme images sampled from the above folder

filenames = random.sample(os.listdir(dirpath), 5000)
for fname in filenames:
    
    srcpath = os.path.join(dirpath, fname)
    print(srcpath)
    destPath = os.path.join(destDirectory, fname)
    print(destPath)
    shutil.copy(srcpath, destPath)