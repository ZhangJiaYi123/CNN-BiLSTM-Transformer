pip install pytorch pytorchvision pytorchaudio numpy pandas scikit-learn tqdm joblib
python preprocess/make_windows.py --csv_dir ./data --out_dir ./outputs --T 50 --stride 25

cd model
python train.py --npz_path ../outputs/cicids2017_windows.npz --checkpoint_dir ../outputs/checkpoints --epochs 30 --batch_size 128
