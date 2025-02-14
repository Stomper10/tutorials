import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv")
valid_data = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv")

ages = train_data['age'].values.astype(np.float16)
min_age, max_age = ages.min(), ages.max()
norm_ages = (ages - min_age) / (max_age - min_age)
train_data["norm_age"] = norm_ages
norm_ages.min()
norm_ages.max()

# EDA train
train_data['age'].describe()
train_data['age'].value_counts().sort_index()

train_data['age'].hist(bins=37)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()
plt.savefig("/shared/s1/lab06/wonyoung/diffusers/sd3/data/age_train.png")
plt.close()

# EDA valid
valid_data['age'].describe()
valid_data['age'].value_counts().sort_index()

valid_data['age'].hist(bins=32)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')
plt.show()
plt.savefig("/shared/s1/lab06/wonyoung/diffusers/sd3/data/age_valid.png")
plt.close()

low = [i for i in range(45, 52)]
high = [i for i in range(74, 82)]
norm_low = (low - min_age) / (max_age - min_age)
norm_high = (high - min_age) / (max_age - min_age)
# 600 each


# low
1+600   # 4 45 600 done 600

17+450  # 3 46 467 done 600

79+450  # 3 47 529 done 530

153+360 # 2 48 513 done 450

259+360 # 2 49 619 done 360

424+180 # 1 50 604 done 180
523+100 # 1 51 623 done 100
# high
511+100 # 1 74 611 done 100

462+150 # 1 75 612 done 150
300+300 # 2 76 600 done 300

207+360 # 2 77 567 done 400

179+360 # 2 78 539 done 430

117+450 # 3 79 567 done 490

57+450  # 3 80 507 done 550

6+600   # 4 81 606 done 600

# 5840 gen

import os
import numpy as np
age_id = "A48"
stage1_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage1"
gen_list = [int(file.split("_")[0][1:]) for file in os.listdir(stage1_dir) if age_id in file]

len(sorted(gen_list))


age_id = "A81"
stage2_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/stage2"
gen_list2 = [int(file.split("_")[1][1:]) for file in os.listdir(stage2_dir) if age_id in file]

len(sorted(gen_list2))


path1="/leelabsg/data/20252_unzip/5503748_20252_2_0/T1/T1.nii.gz"
path2="/leelabsg/data/20252_unzip/3093324_20252_2_0/T1/T1.nii.gz"
600+600+530+450+360+180+100
100+150+300+400+430+490+550+600