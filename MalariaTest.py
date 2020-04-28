import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import pickle          #loading
import smtplib           
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

MY_ADDRESS = 'projectese534@gmail.com'
PASSWORD = 'Qwerty)987'

pkl_filename = "pickle_model.pkl"               #loading
with open(pkl_filename, 'rb') as file:	
    model = pickle.load(file)                   #retrieving the trained model
testImagePath = "malaria/testImages"
imagePaths = list(paths.list_images(testImagePath))   #image paths
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]                       # test images .....loading the image

def read_template(filename):
    """
    Returns a Template object comprising the contents of the 
    file specified by filename.
    """
    
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)

def sendmail(content):
    name = 'patient'

    # read contacts
    if(content[0] == 'Parasitized'):
        message_template = read_template('infected.txt')
    else:
        message_template = read_template('uninfected.txt')
    # set up the SMTP server
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(MY_ADDRESS, PASSWORD)
    email='vigneshc994@gmail.com'
    # For each contact, send the email:
    msg = MIMEMultipart()    
    msgToAuth = MIMEMultipart()   # create a message

        # add in the actual person name to the message template
    message = message_template.substitute(PERSON_NAME=name.title())

        # Prints out the message body for our sake
    print(message)

        # setup the parameters of the message
    msg['From']=MY_ADDRESS
    msg['To']=email
    msg['Subject']="This is TEST"
        
        # add in the message body
    msg.attach(MIMEText(message, 'plain'))
        
        # send the message via the server set up earlier.
    s.send_message(msg)
    del msg
        
    # Terminate the SMTP session and close the connection
    s.quit()

print("loaded images")

# initialize our list of results
results = []

for p in imagePaths:                #for rvery image path reading the images
	# load our original input image
	orig = cv2.imread(p)
 
	# pre-process our image by converting it from BGR to RGB channel
	# ordering (),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]
	image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)        #converting into rgb
	image = cv2.resize(image, (64, 64))                  #resizing
	image = image.astype("float") / 255.0                 #rescaling image intensity----------pixel as 0 & 1
	# order channel dimensions (channels-first or channels-last)
	# depending on our Keras backend, then add a batch dimension to
	# the image
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
 
	# make predictions on the input image
	pred = model.predict(image)                          #prediction
	pred = pred.argmax(axis=1)[0]
 
	# an index of zero is the 'parasitized' label while an index of
	# one is the 'uninfected' label
	label = "Parasitized" if pred == 0 else "Uninfected"
	color = (0, 0, 255) if pred == 0 else (0, 255, 0)

	results.append(label)
print("Result is : ", results)
sendmail(results)      