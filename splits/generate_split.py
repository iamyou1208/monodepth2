import glob
import os
import argparse

parser = argparse.ArgumentParser(description='description')
parser.add_argument("val", type=int, choices=[1, 2, 3, 4, 5])
args = parser.parse_args()

train_less = False

home_dir = os.path.expanduser("~")
file_path = os.path.dirname(os.path.abspath(__file__))
split_carla_dir = os.path.join(file_path, "carla_1k")
carla_dataset_dir = os.path.join(home_dir, "dataset", "depth", "carla_dataset_1k")

towns = ["Town01", "Town02", "Town03", "Town04", "Town05"]
num_towns = len(towns)
num_train_towns = num_towns-2

val_town_id = int(args.val) # Town01 -> 1
val_town_index = val_town_id-1 # Town01 -> 0

if val_town_id == num_towns:
    test_town = towns[0]
else:
    test_town = towns[val_town_index+1]

val_town = towns[val_town_index]

if val_town_index == 0:
    train_towns = towns[2:]
elif val_town_index+2 < num_towns:
    train_towns = towns[(val_town_index+2)%num_towns:] + towns[:(val_town_index)%num_towns]
else:
    train_towns = towns[(val_town_index+2)%num_towns:val_town_index%num_towns]

split = None

split = lambda x: x[-26:-16]
train_files = ""
for train_town in train_towns:
    town_path = os.path.join(carla_dataset_dir, "leftImg8bit", "train", train_town) 
    colors = glob.glob(town_path + "/*.png")
    # print(colors[0]) # ../carla_dataset/leftImg8bit/Town03/Town03_10639_leftImg8bit.png
    # print(town_path)

    # print(colors[0][-26:-16]) # %10d
    common = list(map(split, colors))
    # print(common[0]) # Town03/Town03_10639
    # cnt = 0
    for item in common:
        num = int(item)
        if num == 0 or num == len(common)-1:
            pass
        # elif train_less and cnt==int(len(common)/5):
        #     break
        else:
            train_files += train_town + " " + str(num) + " l" + "\n"
            # cnt += 1

# print(train_files)
with open(split_carla_dir+"/train_files.txt", mode="w") as f:
    f.write(train_files)

val_files = ""
town_path = os.path.join(carla_dataset_dir, "leftImg8bit", "train", val_town) 
# print(town_path)
colors = glob.glob(town_path + "/*.png")
# print(colors[0]) # ../carla_dataset/leftImg8bit/Town03/Town03_10639_leftImg8bit.png
common = list(map(split, colors))
# print(common[0]) # Town03/Town03_10639
# cnt = 0
for item in common:
    num = int(item)
    if num == 0 or num == len(common)-1:
        pass
    # elif train_less and cnt==int(len(common)/5):
    #     break
    else:
        val_files += val_town + " " + str(num) + " l" + "\n"
        # cnt += 1

    # print(val_files)
with open(split_carla_dir+"/val_files.txt", mode="w") as f:
    f.write(val_files)
print(f"val: {val_town}, test: {test_town}, train: {train_towns}")
