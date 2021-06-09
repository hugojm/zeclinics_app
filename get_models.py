
import gdown
urls=['https://drive.google.com/uc?id=1C9xK9saEEolu3lkn57AY0WlejJoArrGB',
      'https://drive.google.com/uc?id=1VKmsaWw9dTFJlHttoU7ebu5P5i-7dKzn',
      'https://drive.google.com/uc?id=1NaN4F9xOerLsWNGV2DZFw21k0UvqdA3n']
output = './zeclinics_app/static/weight/'
for url in urls:
    gdown.download(url,output,quiet=False)
