import random
import scipy.misc
import scipy

# Initializing train and validation or test pointers to 0
train_pointer = 0
test_pointer = 0
split_x = []
split_y = []
#Load data.txt from driving datset
with open("driving_dataset/data.txt") as a:
    for i in a:
        print (i)
        split_x.append("driving_dataset/" + i.split()[0])
        print("driving_datset" + i.split()[0])
        value = float(i.split()[1])
        y_radian = value * scipy.pi /180
        split_y.append(y_radian)
        print(y_radian)

l = len(split_x)*0.8
m = len(split_x)*0.2
#Calculate number of images
no_of_images = len(split_x)

train_split_x = split_x[:int(l)]
num_train_images = len(split_x[:int(l)])
train_split_y = split_y[:int(l)]

val_split_x = split_x[-int(m):]
num_val_images = len(split_x[-int(m):])
val_split_y = split_y[-int(m):]
def LoadTrainData(batch_size):
    global train_pointer
    x_output = []
    y_output = []
    for i in range(0, batch_size):
        j = (train_pointer + i) % num_train_images
        img_read = scipy.misc.imread(train_split_x[j])
        image_read = img_read[-150:]
        resized_img = scipy.misc.imresize(image_read, [66, 200])
        x_output.append(resized_img / 255.0)
        y_output.append([train_split_y[j]])
    train_pointer = train_pointer + batch_size
    return x_output, y_output

def LoadTestData(batch_size):
    global test_pointer
    x_output = []
    y_output = []
    for i in range(0, batch_size):
        c = (test_pointer + i) % num_val_images
        img_read = scipy.misc.imread(val_split_x[c])
        image_read = img_read[-150:]
        resized_img = scipy.misc.imresize(image_read, [66, 200])
        x_output.append(resized_img / 255.0)
        y_output.append([val_split_y[(test_pointer + i) % num_val_images]])
    test_pointer = test_pointer + batch_size
    return x_output, y_output

