# FSPicker

FSPicker is a deep learning-based tool for **Cryo-ET particle picking**. It provides an automated pipeline for preprocessing, training, testing, and evaluating 3D particle picking models.

## **Features**

- **Preprocessing**: Convert raw data into a format suitable for deep learning.
- **Configurable Training Pipeline**: Users can customize training parameters.
- **Deep Learning Models**: Supports models like ResUNet, and DSMSPNet.
- **Testing & Evaluation**: Provides automated testing and metric evaluation.

---

## **Installation & Environment Setup**

You need to set up a **Python 3.8.3** environment using `conda`:

```bash
# Create a new environment
conda create -n fspicker -c conda-forge python=3.8.3 -y

# Alternatively, specify the environment path
conda create --prefix /path/to/your/conda/env python=3.8.3 -y

# Activate the environment
conda activate /path/to/your/conda/env
```

---

## **Usage Guide**

### **1. Preprocessing Data**

Convert raw input data into the correct format for training:

```bash
cd work/FSPicker/bin

python preprocess.py --pre_configs '/path/to/preprocess/config1.py'

python preprocess.py --pre_configs '/path/to/preprocess/config2.py'
```

---

### **2. Generating Training Configuration**

Configure training parameters and save them to a configuration file:

```bash
python generate_train_config.py --pre_configs '/path/to/preprocess/config1.py'     --dset_name 'SHREC_2021_train'     --cfg_save_path '/path/to/save/configs/'     --train_set_ids '0-7'     --val_set_ids '8'     --batch_size 8     --block_size 72     --pad_size 12     --learning_rate 0.001     --max_epoch 100     --threshold 0.5     --gpu_id 0
```

---

### **3. Training the Model**

Once the training configuration file is ready, train the model:

```bash
python train_bash.py --train_configs '/path/to/training/config.py'
```

---

### **4. Testing the Model**

Evaluate the trained model using a pre-trained checkpoint:

```bash
python test_bash.py --train_configs '/path/to/training/config.py'     --checkpoints '/path/to/checkpoints/model_epoch_3.ckpt'
```

## **License**

FSPicker is released under the **MIT License**.

---

## **Acknowledgements**

We thank contributors and researchers who have contributed to Cryo-ET particle picking.

---

🚀 **Enjoy using FSPicker for your Cryo-ET research!**
