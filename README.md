**Siamese Topographic Generation Model (STGM)**
a deep learning model designed to reconstruct high-resolution and high-precision subglacial topography maps of Antarctica using sparse radio-echo sounding (RES) data.

**Dependencies**
Python 3.7
PyTorch >= 1.0
opencv-python==3.4.17.63
pyyaml==5.3.1
pillow==9.4.0
lpips==0.1.4
**Modify Configuration:**
Open checkpoints/config.yml and adjust the parameters according to your needs.
Modify the train_data_list and test_data_list in data/data_list to point to your dataset directories.
**Dataset**
The project utilizes ArcticDEM data for training due to the lack of comprehensive Antarctic subglacial topography data. The ArcticDEM data is available at https://www.pgc.umn.edu/data/arcticdem/.
