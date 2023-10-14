import os

train_val_scenes_content = '''Potsdam_ISPRS/top_potsdam_2_10_RGB
Potsdam_ISPRS/top_potsdam_2_11_RGB
Potsdam_ISPRS/top_potsdam_2_12_RGB
Potsdam_ISPRS/top_potsdam_2_13_RGB
Potsdam_ISPRS/top_potsdam_2_14_RGB
Potsdam_ISPRS/top_potsdam_3_10_RGB
Potsdam_ISPRS/top_potsdam_3_11_RGB
Potsdam_ISPRS/top_potsdam_3_13_RGB
Potsdam_ISPRS/top_potsdam_4_10_RGB
Potsdam_ISPRS/top_potsdam_5_10_RGB
Potsdam_ISPRS/top_potsdam_6_7_RGB
Potsdam_ISPRS/top_potsdam_6_8_RGB
Potsdam_ISPRS/top_potsdam_6_9_RGB
Selwyn_LINZ/Selwyn_BX22_Tile_LEFT_15cm_0004
Selwyn_LINZ/Selwyn_BX22_Tile_RIGHT_15cm_0001
Selwyn_LINZ/Selwyn_BX22_Tile_RIGHT_15cm_0003
Toronto_ISPRS/03553
Toronto_ISPRS/03559
Toronto_ISPRS/03747
'''

test_scenes_content = '''Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_31.99319028-Oct-2007_11-00-31.993_Frame_1-124%
Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_42.40561128-Oct-2007_11-00-47.194_Frame_74-124%
Columbus_CSUAV_AFRL/EO_Run01_s2_301_15_00_52.82681728-Oct-2007_11-01-01.775_Frame_144-124%
Utah_AGRC/12TVL180140
Utah_AGRC/12TVL200180
Utah_AGRC/12TVL240120
Utah_AGRC/12TVK220980-CROP
Utah_AGRC/12TVL160640-CROP
Utah_AGRC/12TVL160660-CROP
Utah_AGRC/12TVL220360-CROP
Vaihingen_ISPRS/TOP_Mosaic_09cm_scaled_15cm_Gray
'''

def create_nested_folder(out_folder = '/content/cowc_processed'):
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)

  train_val_folder = os.path.join(out_folder, 'train_val')
  test_folder = os.path.join(out_folder, 'test')

  if not os.path.exists(test_folder):
    os.makedirs(train_val_folder)

  if not os.path.exists(test_folder):
    os.makedirs(test_folder)

  # Create train_val_scenes.txt & test_scenes.txt
  with open(os.path.join(train_val_folder, 'train_val_scenes.txt'), 'w') as f:
    f.write(train_val_scenes_content)
  f.close()

  with open(os.path.join(test_folder, 'test_scenes.txt'), 'w') as f:
    f.write(test_scenes_content)
  f.close()