# chatbot

## Working
Simple implementation of an LRL neural net based classifier using the bag of words idea as word vectors. Based on the type of request identified, we then choose a random (related) answer to make it work as a chatbot.
This project is based on the ideas presented in this tutorial: https://www.youtube.com/watch?v=RpWeNzfSUHw. Thanks to Patrick Loeber for coming up with such a great tutorial.

## Setup
To setup conda environment (on windows):
```bash
conda create --name chatbot --file requirements.txt
conda activate chatbot
```
To train the bot
```bash
python training.py
```
To test the bot
```bash
python bot.py
```
Feel free to change the training data in data.json
