# PyTorch Flask MultiTag Vacation Dataset Classifier

Check the demo hosted on heroku [here](https://multitagflask.herokuapp.com/) (it may be a bit slow to load since it is free-tier)

An image scraper version of the Flask app is located on the flaskv2 branch.

Summer project intented to classify and tag a dataset of tens of thousands of vacation photos taken by my family members over a number of years. The model uses resnet50 as a base and via transfer learning utilizes a hand labeled dataset of about 1500 vacation photos to be better biased towards the inputs. I used a skeleton code for [multitag pytorch training](https://debuggercafe.com/multi-label-image-classification-with-pytorch-and-deep-learning/) as well as a [flask-heroku](https://github.com/avinassh/pytorch-flask-api-heroku) web app skeleton code and modified it to suit my needs. I am still tuning the model's accuracy so further commits updating 'model.pth' will occur on a regular basis.

# Flowchart

![Multitag Flowchart](https://i.imgur.com/vlNSvOn.png)

## Requirements

Install them from `requirements.txt`:

    pip install -r requirements.txt


## Local Deployment

Run the server:

    python app.py


## Heroku Deployment

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/terencelee-uni/multitag)


## License

The mighty MIT license. Please check `LICENSE` for more details.
