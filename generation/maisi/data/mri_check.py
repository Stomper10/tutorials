import nibabel as nib
import matplotlib.pyplot as plt

# Load the NIfTI file
file_name = "T1_orig_defaced.nii.gz"
file_path = f'/leelabsg/data/20252_unzip/1000502_20252_2_0/T1/{file_name}'
img = nib.load(file_path)
data = img.get_fdata()

# Select a slice to visualize (e.g., the middle slice along the z-axis)
slice_idx = data.shape[2] // 2  # Adjust for another axis if needed
plt.imshow(data[:, slice_idx, :], cmap='gray')
# plt.title(f'Slice {slice_idx}')
# plt.axis('off')
# plt.show()
plt.savefig(f"/shared/s1/lab06/wonyoung/diffusers/sd3/data/{file_name}.png")


#########################

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the input files
mri_pheno_1_path = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/raw/MRI_pheno_2_T1image.csv"
mri_pheno_2_path = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/raw/MRI_Pheno_2_T1image2.csv"
ukbb_cn_region_path = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/raw/ukbb_cn_region.csv"
real_data_list = [int(filename[:7]) for filename in os.listdir("/leelabsg/data/20252_unzip") if filename[-3] == "2"]

# Read the data
mri_pheno_1 = pd.read_csv(mri_pheno_1_path)
mri_pheno_2 = pd.read_csv(mri_pheno_2_path)
mri_pheno = pd.merge(
    mri_pheno_1,
    mri_pheno_2,
    on="eid",
    how="inner"
)
ukbb_cn_region = pd.read_csv(ukbb_cn_region_path)
ukbb_cn_region = ukbb_cn_region[["subjectID", "age", "sex"]]
merge_data = pd.merge(
    mri_pheno,
    ukbb_cn_region.rename(columns={"subjectID": "eid"}),
    on="eid",
    how="inner"
)
filtered_merge_data = merge_data[merge_data['eid'].isin(real_data_list)]

md_filtered = filtered_merge_data[filtered_merge_data.columns[~filtered_merge_data.columns.str.contains('i3')]]
md_filtered['rel_path'] = md_filtered['eid'].apply(lambda eid: f"{eid}_20252_2_0/T1/T1_brain_to_MNI.nii.gz")

new_column_order = ['eid', 'age', 'sex',
                    'p25001_i2', 'p25002_i2', 'p25003_i2', 'p25004_i2', 'p25005_i2',
                    'p25006_i2', 'p25007_i2', 'p25008_i2', 'p25009_i2', 'p25010_i2', 'p25025_i2',
                    'rel_path']
final_df = md_filtered[new_column_order].sort_values('eid').reset_index(drop=True)

train_df, test_df = train_test_split(final_df, test_size=0.1, random_state=42)
# train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
# valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

train_df.to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv", index=False) # 22726
train_df[:100].to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_small.csv", index=False)
test_df.to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv", index=False) # 2526
test_df[:20].to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_small.csv", index=False)
# train_df.to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_cls.csv", index=False) # 22726
# train_df[:100].to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/train_cls_small.csv", index=False)
# valid_df.to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_cls.csv", index=False) # 2526
# valid_df[:20].to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid_cls_small.csv", index=False)
# test_df.to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/test_cls.csv", index=False) # 2526
# test_df[:20].to_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/test_cls_small.csv", index=False)



# RAW generation for quality comparison
import os
import torch
import nibabel as nib
from PIL import Image
from monai import transforms
import matplotlib.pyplot as plt

def save_slices_as_images(volume_data):
    slice_images = []
    for i in range(volume_data.shape[0]): ### c:0, s:1, a:2
        plt.figure(figsize=(7, 7))
        plt.imshow(volume_data[i, :, :], cmap='gray') ### c, s, a
        plt.title(f'Slice {i}')
        plt.axis('off')
        
        # Save each slice to a temporary file
        file_name = f'tmp_slices/slice_{i}.png'
        plt.savefig(file_name)
        plt.close()
        
        # Open the image and append to the list
        slice_images.append(Image.open(file_name))
    
    return slice_images

raw_path = "/leelabsg/data/20252_unzip/1000502_20252_2_0/T1/T1_brain_to_MNI.nii.gz"
raw_img = nib.load(raw_path)
raw_img = raw_img.get_fdata()
raw_img = torch.from_numpy(raw_img) # Stays on CPU # (182, 218, 182)
axes_mapping = {
    's': (0, 1, 2),
    'c': (1, 0, 2),
    'a': (2, 1, 0)
}

raw_img = raw_img.permute(*axes_mapping["c"]).unsqueeze(0) # (1, 218, 182, 182)

img_dict = dict()
img_dict["pixel_values"] = raw_img.to(torch.float16)

input_size = (224,40,40) # SET (224,160,160)
train_transforms = transforms.Compose(
        [
            transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
        ]
    )
sample = train_transforms(img_dict)

output_dir_gif="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/gen_volumes/gif" # SET /stage1
os.makedirs('tmp_slices', exist_ok=True)
gen_volume_data = sample["pixel_values"].squeeze().numpy()
gen_slice_images = save_slices_as_images(gen_volume_data)
gen_slice_images[0].save(f"{output_dir_gif}/raw_stage1_160.gif", save_all=True, append_images=gen_slice_images[1:], duration=200, loop=0)

# Cleanup the temporary image files
for img_file in os.listdir('tmp_slices'):
    os.remove(os.path.join('tmp_slices', img_file))
os.rmdir('tmp_slices')
print(f"GIF saved as {output_dir_gif}/raw_stage1_160.gif")
