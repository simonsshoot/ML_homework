#----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中,通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from frcnn import FRCNN


if __name__ == "__main__":
    frcnn = FRCNN()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测,如果想对预测过程进行修改,如保存图片,截取对象等,可以先看下方详细的注释
    #   'video'             表示视频检测,可调用摄像头或者视频进行检测,详情查看下方注释。
    #   'fps'               表示测试fps,使用的图片是img里面的street.jpg,详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹,保存img_out文件夹,详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径,当video_path=0时表示检测摄像头
    #                       想要检测视频,则设置如video_path = "xxx.mp4"即可,代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径,当video_save_path=""时表示不保存
    #                       想要保存视频,则设置如video_save_path = "yyy.mp4"即可,代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候,图片检测的次数。理论上test_interval越大,fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    VOCdevkit_path="./elec_data"
    '''
        1、该代码无法直接进行批量预测,如果想要批量预测,可以利用os.listdir()遍历文件夹,利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py,在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存,利用r_image.save("img.jpg")即可保存,直接在predict.py里进行修改即可。 
        3、如果想要获得预测框的坐标,可以进入frcnn.detect_image函数,在绘图部分读取top,left,bottom,right这四个值。
        4、如果想要利用预测框截取下目标,可以进入frcnn.detect_image函数,在绘图部分利用获取到的top,left,bottom,right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字,比如检测到的特定目标的数量,可以进入frcnn.detect_image函数,在绘图部分对predicted_class进行判断,
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车,然后记录数量即可。利用draw.text即可写字。
    '''
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                print('Predicting...')
                #修改，原始为：
                # r_image = frcnn.detect_image(image, crop = crop, count = count)
                # print('Done.')
                # r_image.show()
                r_image = frcnn.detect_image(image, crop = crop, count = count)
                output_dir="./output"
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img))[0]
                filename=f"{base_name}.result.jpg"
                output_path = os.path.join(output_dir, filename)
                r_image.save(output_path, quality=95)
                print('Done.')

    #ps：上面已经更改实现了单个图片的预测
    # if mode == "predict":
    #     test_txt_path    = os.path.join(VOCdevkit_path, "ImageSets/Main/test.txt")
    #     jpeg_images_dir  = os.path.join(VOCdevkit_path, "JPEGImages")        
    #     output_dir       = os.path.abspath("./output/elec_data/")                          
    #     allowed_exts     = ['.jpg', '.jpeg', '.png']                              
    #     os.makedirs(output_dir, exist_ok=True)
    #     with open(test_txt_path, "r", encoding='utf-8') as f:
    #         base_names = [line.strip() for line in f.readlines()]

    #     for base_name in base_names:
    #         found = False
    #         for ext in allowed_exts:
    #             image_path = os.path.join(jpeg_images_dir, f"{base_name}{ext}")
    #             if os.path.exists(image_path):
    #                 found = True
    #                 break
            
    #         if not found:
    #             print(f"{base_name} not found, allowed extensions：{allowed_exts}")
    #             continue
    #         try:
    #             image = Image.open(image_path)
    #             print(f"predicting：{base_name}{ext}")
                
    #             r_image = frcnn.detect_image(image, crop=crop, count=count)

    #             output_path = os.path.join(output_dir, f"{base_name}.result.jpg")
    #             r_image.save(output_path, quality=95, subsampling=0)
    #             print(f"save to：{output_path}\n")
                
    #         except Exception as e:
    #             print(f"handle {base_name} error: {str(e)}")
    #             continue

    #     print(f"batch predict done, total {len(base_names)} files, result saved in {output_dir}")

                
    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref,frame = capture.read()
            # 格式转变,BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(frcnn.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps  = ( fps + (1. / (time.time() - t1)) ) / 2
            print("fps = %.2f"%(fps))
            frame = cv2.putText(frame, "fps = %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff 
            if video_save_path != "":
                out.write(frame)
            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()


    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')


    elif mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = frcnn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality = 95, subsampling = 0)


    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
