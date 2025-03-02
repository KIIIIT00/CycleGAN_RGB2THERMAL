import os
import cv2
from PIL import Image
import numpy as np
from options.camera_test_options import CameraTestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
    
def capture_and_display_frame(model, opt):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # モデルに入力
        data = {'A': frame_pil}
        model.set_input(data)
        model.test()

        # 処理結果を取得し、OpenCVで表示できる形式に変換
        visuals = model.get_current_visuals()
        processed_image = visuals['fake_B']  # 出力画像が'fake_B'の場合
        processed_image = np.array(processed_image)  # PIL画像をNumPy配列に変換
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)  # OpenCV形式に変換

        # カメラの元画像と処理後の画像を並べて表示
        combined_image = cv2.hconcat([frame, processed_image])  # 左: 元画像, 右: 処理後画像
        cv2.imshow('Camera Feed - Original and Processed', combined_image)

        # キー入力待ち。'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = CameraTestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    model = create_model(opt)
    model.setup(opt)

    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run

    if opt.eval:
        model.eval()
    
    capture_and_display_frame(model, opt)